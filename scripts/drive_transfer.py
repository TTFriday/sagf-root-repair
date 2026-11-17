"""Train a binary U-Net and evaluate anchored reconstruction on DRIVE."""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from skimage.morphology import skeletonize

from toposeg.metrics import cldice_score, tolerant_point_metrics, _skeleton_points
from toposeg.phenotyping import skeleton_points
from toposeg.reconstruction import ReconstructionParams, reconstruct

DATA = Path(os.environ.get("TOPOSEG_DRIVE_ROOT", "../data/09_image_retinal_DRIVE"))


class SmallUNet(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.e1, self.e2, self.e3 = block(1, base), block(base, base * 2), block(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d2, self.d1 = block(base * 4, base * 2), block(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        d2 = self.d2(torch.cat([self.up2(e3), e2], 1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)


def load_split(split):
    items = []
    for img_path in sorted((DATA / split / "input").iterdir()):
        lab_path = DATA / split / "label" / (img_path.stem.replace("_training", "_manual1").replace("_test", "_manual1") + ".png")
        if not lab_path.is_file():
            cand = list((DATA / split / "label").glob(img_path.stem.split("_")[0] + "*"))
            lab_path = cand[0]
        image = np.asarray(Image.open(img_path).convert("L"), dtype=np.float32) / 255.0
        label = (np.asarray(Image.open(lab_path).convert("L")) > 127).astype(np.uint8)
        items.append((img_path.stem, image, label))
    return items


def dice_loss(logits, target):
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    return (1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)).mean()


def train_model(device, epochs, seed=17):
    """Seed new training only; an existing checkpoint's initialization seed is unknown."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed)
    items = load_split("train")
    model = SmallUNet().to(device)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    patches = []
    for _, image, label in items:
        h, w = image.shape
        for y in range(0, h - 256 + 1, 128):
            for x in range(0, w - 256 + 1, 128):
                patches.append((image[y:y + 256, x:x + 256], label[y:y + 256, x:x + 256]))
    steps = 0
    for epoch in range(epochs):
        rng.shuffle(patches)
        for i in range(0, len(patches) - 7, 8):
            batch = patches[i:i + 8]
            x = torch.from_numpy(np.stack([p[0] for p in batch])[:, None]).to(device)
            y = torch.from_numpy(np.stack([p[1] for p in batch])[:, None]).float().to(device)
            if rng.random() < 0.5:
                x, y = torch.flip(x, [-1]), torch.flip(y, [-1])
            if rng.random() < 0.5:
                x, y = torch.rot90(x, 1, [-2, -1]), torch.rot90(y, 1, [-2, -1])
            logits = model(x)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y) + dice_loss(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            steps += 1
        if (epoch + 1) % 20 == 0:
            print(f"epoch {epoch + 1} loss {loss.item():.4f} steps {steps}", flush=True)
    return model


def vessel_metrics(pred, gt):
    endpoints_p, branches_p = skeleton_points(pred)
    endpoints_g, branches_g = skeleton_points(gt)
    sk_p, sk_g = skeletonize(pred.astype(bool)), skeletonize(gt.astype(bool))
    ep_p, bp_p = _skeleton_points(sk_p)
    ep_g, bp_g = _skeleton_points(sk_g)
    point_e = tolerant_point_metrics(ep_p, ep_g, tolerance=2.0)
    point_b = tolerant_point_metrics(bp_p, bp_g, tolerance=2.0)
    inter = (pred & gt).sum()
    dice = 2 * inter / max(1, pred.sum() + gt.sum())
    return {"dice": float(dice), "cldice": float(cldice_score(pred, gt)),
            "pred_endpoints": endpoints_p, "gt_endpoints": endpoints_g,
            "endpoint_error": endpoints_p - endpoints_g,
            "pred_branches": branches_p, "gt_branches": branches_g,
            "branch_error": branches_p - branches_g,
            "endpoint_f1": point_e["f1"], "branchpoint_f1": point_b["f1"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=17,
                        help="NumPy/PyTorch seed for new training only; does not identify a cached checkpoint's initialization seed")
    parser.add_argument("--output", type=Path, default=Path("outputs/drive_transfer"),
                        help="Directory for the local checkpoint, parameters and numerical CSVs; no PNGs are generated")
    parser.add_argument("--frozen", type=Path, default=Path("outputs/reconstruction/frozen.json"))
    parser.add_argument("--radii", nargs="+", type=int, default=[8, 4, 16])
    parser.add_argument("--data", type=Path, help="DRIVE root (default: TOPOSEG_DRIVE_ROOT or ../data/09_image_retinal_DRIVE)")
    args = parser.parse_args()
    global DATA
    if args.data:
        DATA = args.data
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()
    if (args.output / "unet_drive.pt").is_file():
        model = SmallUNet().to(device)
        model.load_state_dict(torch.load(args.output / "unet_drive.pt", map_location=device, weights_only=True))
        print("loaded cached checkpoint; initialization seed is not inferred from --seed", flush=True)
    else:
        model = train_model(device, args.epochs, seed=args.seed)
        torch.save(model.state_dict(), args.output / "unet_drive.pt")
        print(f"trained with seed {args.seed} in {time.time() - started:.0f}s", flush=True)
    frozen = json.loads(args.frozen.read_text(encoding="utf-8"))
    base = ReconstructionParams(**frozen["selected"]["params"])
    arms = {"raw": ReconstructionParams()}
    for radius in args.radii:
        arms[f"sagf_r{radius}"] = replace(base, forest_anchor="largest", forest_radius=radius,
                                          min_area=1, forest_keep_area=40)
    (args.output / "params.json").write_text(json.dumps({k: asdict(v) for k, v in arms.items()}, indent=1), encoding="utf-8")
    rows = []
    model.eval()
    for name, image, gt in load_split("val"):
        with torch.no_grad():
            h, w = image.shape
            ph, pw = (h + 3) // 4 * 4, (w + 3) // 4 * 4
            padded = np.zeros((ph, pw), dtype=np.float32)
            padded[:h, :w] = image
            prob = torch.sigmoid(model(torch.from_numpy(padded[None, None]).to(device)))[0, 0].cpu().numpy()[:h, :w]
        quantised = np.zeros((6,) + image.shape, dtype=np.uint8)
        quantised[0] = np.rint(prob * 255).astype(np.uint8)
        for arm, params in arms.items():
            mask = reconstruct(quantised, params).astype(bool)
            row = {"image_id": name, "arm": arm, **vessel_metrics(mask, gt.astype(bool))}
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "per_image.csv", index=False)
    summary = frame.groupby("arm").agg(
        dice=("dice", "mean"), cldice=("cldice", "mean"),
        endpoint_mae=("endpoint_error", lambda x: np.abs(x).mean()),
        endpoint_bias=("endpoint_error", "mean"),
        branch_mae=("branch_error", lambda x: np.abs(x).mean()),
        branch_bias=("branch_error", "mean"),
        endpoint_f1=("endpoint_f1", "mean")).reset_index()
    summary.to_csv(args.output / "summary.csv", index=False)
    print(summary.round(4).to_string(index=False), flush=True)
    print(f"total {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()