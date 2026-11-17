from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from toposeg.dataset import SeedlingDataset
from toposeg.engine import load_checkpoint
from toposeg.inference import sliding_window_inference
from toposeg.models import CompactUNet
from toposeg.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Store quantised class probabilities for the OOD split")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    experiment = config.get("experiment", args.config.stem)
    run_dir = Path("outputs") / experiment / f"seed_{args.seed}"
    model = CompactUNet(base_channels=config["model"].get("base_channels", 24), skeleton_head=config["model"].get("skeleton_head", True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_checkpoint(run_dir / "best.pt", model, device=device)
    split_csv = Path(config["data"]["split_csv"]).with_name("ood.csv")
    dataset = SeedlingDataset(str(split_csv), "ood", training=False)
    out_dir = run_dir / "probs_ood"
    out_dir.mkdir(parents=True, exist_ok=True)
    count = len(dataset) if not args.limit else min(args.limit, len(dataset))
    started = time.time()
    for index in range(count):
        image_id = str(dataset.frame.iloc[index].image_id)
        target = out_dir / f"{image_id}.npz"
        if target.is_file():
            continue
        sample = dataset.full_image(index)
        _, probabilities = sliding_window_inference(model, sample["image"], config["inference"]["patch_size"], config["inference"].get("overlap", 0.5), device=device)
        quantised = np.rint(probabilities[1:].numpy() * 255).astype(np.uint8)
        tmp = target.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, p=quantised)
        tmp.replace(target)
    print(f"{run_dir} ood {count} images {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()