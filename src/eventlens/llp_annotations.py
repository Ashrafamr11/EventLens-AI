
import csv
from pathlib import Path


SPLIT_FILES = {
    "validation": "AVVP_val_pd.csv",
    "test": "AVVP_test_pd.csv",
}

ANNOTATION_FILES = {
    "audio": "AVVP_eval_audio.csv",
    "visual": "AVVP_eval_visual.csv",
}


def load_temporal_annotations(
    split: str,
    modality: str,
    data_dir: str | Path = "data/raw/llp",
) -> list[dict]:

    if split not in SPLIT_FILES:
        raise ValueError("Split must be validation or test.")

    if modality not in ANNOTATION_FILES:
        raise ValueError("Modality must be audio or visual.")

    data_dir = Path(data_dir)

    with (data_dir / SPLIT_FILES[split]).open(
        encoding="utf-8-sig", newline=""
    ) as file:
        split_ids = {
            row["filename"]
            for row in csv.DictReader(file, delimiter="\t")
        }

    events = []

    with (data_dir / ANNOTATION_FILES[modality]).open(
        encoding="utf-8-sig", newline=""
    ) as file:
        for row in csv.DictReader(file, delimiter="\t"):
            if row["filename"] not in split_ids:
                continue

            events.append({
                "filename": row["filename"],
                "label": row["event_labels"],
                "start_sec": int(row["onset"]),
                "end_sec": int(row["offset"]),
                "modality": modality,
            })

    return events
