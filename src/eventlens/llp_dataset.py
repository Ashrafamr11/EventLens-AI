
import csv
from pathlib import Path

SPLIT_FILES = {
    "train": "AVVP_train.csv",
    "validation": "AVVP_val_pd.csv",
    "test": "AVVP_test_pd.csv",
}


def load_clip_labels(
    split: str,
    data_dir: str | Path = "data/raw/llp",
) -> list[dict]:
    if split not in SPLIT_FILES:
        raise ValueError("Invalid dataset split.")

    csv_path = Path(data_dir) / SPLIT_FILES[split]

    samples = []

    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")

        for row in reader:
            labels = [
                label.strip()
                for label in row["event_labels"].split(",")
                if label.strip()
            ]

            samples.append({
                "filename": row["filename"],
                "video_id": row["filename"][:11],
                "labels": labels,
            })

    return samples
