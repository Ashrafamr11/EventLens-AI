"""Extract only LLP train + validation arrays from saved original feature ZIP.

Reads a checkpoint-free media-feature archive on Google Drive; copies only the
required .npy files and small manifests to the temporary Colab workspace.
Never writes into or changes the saved original archive.
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from zipfile import ZipFile

MODALITIES = ("vggish", "res152", "r2plus1d_18")
SPLITS = {"train": ("AVVP_train.csv", 10000), "validation": ("AVVP_val_pd.csv", 649)}


def read_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        first_line = f.readline()
        f.seek(0)
        rows = csv.DictReader(f, delimiter="\t" if "\t" in first_line else ",")
        if "filename" not in (rows.fieldnames or []):
            raise ValueError(f"Expected filename column in {path}")
        ids = [row["filename"].strip()[:11] for row in rows if row.get("filename")]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs in {path}")
    return ids


def prepare(artifact_root: Path, project_root: Path) -> dict[str, int]:
    local_data = project_root / "data/raw/llp"
    drive_data = artifact_root / "data/raw/llp"
    local_data.mkdir(parents=True, exist_ok=True)

    filenames = [s[0] for s in SPLITS.values()] + ["AVVP_eval_audio.csv", "AVVP_eval_visual.csv"]
    for fn in filenames:
        dst = local_data / fn
        if not dst.is_file():
            src = drive_data / fn
            if not src.is_file():
                raise FileNotFoundError(f"Missing LLP manifest: {src}")
            shutil.copyfile(src, dst)

    split_ids = {}
    for split, (filename, size) in SPLITS.items():
        ids = read_ids(local_data / filename)
        if len(ids) != size:
            raise ValueError(f"{split}: expected {size} clips, got {len(ids)}")
        split_ids[split] = ids

    if set(split_ids["train"]) & set(split_ids["validation"]):
        raise ValueError("Data leakage: train and validation contain overlapping IDs")

    feat_dir = project_root / "data/processed/feats"
    requested = [(mod, video_id) for ids in split_ids.values() for video_id in ids for mod in MODALITIES]
    missing = [(mod, ident) for mod, ident in requested if not (feat_dir/mod/f"{ident}.npy").is_file()]
    print(f"ASSETS: {len(split_ids['train'])} train, {len(split_ids['validation'])} val, {len(missing)} feature arrays missing", flush=True)
    if not missing:
        print("TRAINING ASSETS READY", flush=True)
        return {"train": len(split_ids["train"]), "validation": len(split_ids["validation"])}

    source = artifact_root / "data/feats.zip"
    if not source.is_file():
        raise FileNotFoundError(f"Original LLP features archive missing: {source}")

    with ZipFile(source) as z:
        lookup = {}
        for name in z.namelist():
            parts = name.replace("\\", "/").split("/")
            if len(parts) < 2 or parts[-2] not in MODALITIES or not parts[-1].endswith(".npy"):
                continue
            key = (parts[-2], parts[-1][:-4])
            if key in lookup:
                raise ValueError(f"Duplicate entry in saved feature archive: {key}")
            lookup[key] = name
        absent = [f"{mod}/{ident}.npy" for mod, ident in missing if (mod, ident) not in lookup]
        if absent:
            raise FileNotFoundError(f"Missing feature arrays inside the archive: {absent[:5]}")
        expected_bytes = sum(z.getinfo(lookup[key]).file_size for key in missing)
        free_bytes = shutil.disk_usage(project_root).free
        if free_bytes < expected_bytes + (2 * 1024 ** 3):
            raise RuntimeError(f"Not enough free disk space: need {(expected_bytes + 2*1024**3)/1024**3:.1f} GiB; have {free_bytes/1024**3:.1f} GiB")
        for i, (mod, ident) in enumerate(missing, 1):
            dest = feat_dir/mod/f"{ident}.npy"
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(lookup[(mod, ident)]) as src, dest.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            if i % 2500 == 0 or i == len(missing):
                print(f"Restored {i}/{len(missing)} feature arrays", flush=True)

    print("TRAINING ASSETS READY", flush=True)
    return {"train": len(split_ids["train"]), "validation": len(split_ids["validation"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.artifact_root.expanduser().resolve(), args.project_root.expanduser().resolve())
