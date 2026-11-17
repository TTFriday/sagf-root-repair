import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from toposeg.manifest import build_manifest, parse_folder_metadata
from toposeg.splits import make_group_splits


def test_manifest_event_parsing_and_build(tmp_path: Path):
    folder_a = "Manual_Annotation_ArabidopsisDataset_Roots_rpi14_2020-03-12_17-00_1"
    folder_b = "Manual_Annotation_ArabidopsisDataset_Roots_rpi14_2020-03-12_17-00_4"
    parsed_a, parsed_b = parse_folder_metadata(folder_a), parse_folder_metadata(folder_b)
    assert parsed_a["event_id"] == parsed_b["event_id"]
    assert parsed_a["condition"] == "Roots"
    assert parsed_a["rpi"] == "rpi14"
    assert parsed_a["camera"] == "1"
    assert parse_folder_metadata("x_ArabidopsisDataset_Germination_MultipleVids_Varios")["is_multiple_vids"]

    (tmp_path / "imagesTr").mkdir()
    (tmp_path / "labelsTr").mkdir()
    mapping = {folder_a: ["image_1"], folder_b: ["image_2"]}
    (tmp_path / "folder_image_map.json").write_text(json.dumps(mapping), encoding="utf-8")
    for index in (1, 2, 3):
        Image.fromarray(np.full((8, 9), index, np.uint8)).save(tmp_path / "imagesTr" / f"image_{index}_0000.png")
        label = np.zeros((8, 9), np.uint8)
        label[2:4, 2:4] = index if index < 3 else 1
        Image.fromarray(label).save(tmp_path / "labelsTr" / f"image_{index}.png")
    frame = build_manifest(tmp_path, expected_count=None, compute_hashes=False)
    assert len(frame) == 3
    assert frame.image_id.is_unique
    assert frame.mapped.sum() == 2
    assert frame.loc[frame.image_id == "image_3", "condition"].item() == "unmapped"


def test_group_split_has_no_event_leakage():
    rows = []
    for event in range(30):
        for image in range(2):
            rows.append({
                "image_id": f"image_{event}_{image}", "event_id": f"event_{event}",
                "condition": "A" if event % 2 else "B", "mapped": event != 29,
                "image_sha256": f"ih-{event}-{image}", "label_sha256": f"lh-{event}-{image}",
                **{f"has_class_{c}": int((event + c) % 3 != 0) for c in range(1, 7)},
            })
    source = pd.DataFrame(rows)
    main_a, ood_a = make_group_splits(source, seed=20261016)
    main_b, _ = make_group_splits(source, seed=20261016)
    assert main_a[["image_id", "split"]].equals(main_b[["image_id", "split"]])
    assert main_a.groupby("event_id").split.nunique().max() == 1
    assert set(main_a.split) == {"train", "val", "test"}
    assert set(main_a.image_id).isdisjoint(set(ood_a.image_id))
    assert len(ood_a) == 2


def test_selection_verifies_relative_labels_under_external_data_root(tmp_path, monkeypatch):
    import scripts.select_reconstruction as selection

    data_root = tmp_path / "external_data"
    (data_root / "labelsTr").mkdir(parents=True)
    label = data_root / "labelsTr/image.png"
    Image.fromarray(np.ones((4, 4), dtype=np.uint8)).save(label)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([{"image_id": "image", "image_path": "imagesTr/image.png",
                   "label_path": "labelsTr/image.png", "label_sha256": selection.sha256(label)}]).to_csv(manifest, index=False)
    monkeypatch.setenv("TOPOSEG_DATA_ROOT", str(data_root))
    monkeypatch.setattr(selection, "MANIFEST", manifest)
    run = tmp_path / "baseline_region/seed_17"
    (run / "probs_val").mkdir(parents=True)
    probability = run / "probs_val/image.npz"
    np.savez(probability, p=np.zeros((6, 4, 4), dtype=np.uint8))
    result = tmp_path / "validation.csv"
    pd.DataFrame([{"run": str(run), "image_id": "image", "event_id": "event", "candidate": "raw",
                   **{metric: 0.5 for metric in selection.REPORT}, "root_asd_px": 1.0}]).to_csv(result, index=False)
    protocol = {"schema_version": 1, "split": "val", "runs": [str(run)],
                "images": [{"image_id": "image", "event_id": "event"}], "grid": {"raw": {}},
                "manifest_sha256": selection.sha256(manifest), "label_sha256": {"image": selection.sha256(label)},
                "probability_sha256": {str(run): {"image": selection.sha256(probability)}},
                "implementation_sha256": {name: selection.sha256(Path(name)) for name in selection.EVALUATION_IMPLEMENTATIONS}}
    selection.protocol_path(result).write_text(json.dumps(protocol), encoding="utf-8")
    assert len(selection.load_results([result])) == 1
