
import numpy as np


def build_temporal_targets(events, class_names):
    num_segments = 10
    class_to_index = {
        name: i for i, name in enumerate(class_names)
    }

    targets = {
        "audio": np.zeros((10, len(class_names)), dtype=np.float32),
        "visual": np.zeros((10, len(class_names)), dtype=np.float32),
    }

    for event in events:
        modality = event["modality"]
        label = event["label"]
        start = event["start_sec"]
        end = event["end_sec"]

        if modality not in targets:
            raise ValueError(f"Unknown modality: {modality}")

        if label not in class_to_index:
            raise ValueError(f"Unknown label: {label}")

        if not (0 <= start < end <= num_segments):
            raise ValueError("Invalid temporal annotation.")

        class_index = class_to_index[label]

        targets[modality][start:end, class_index] = 1.0

    targets["joint"] = np.maximum(
        targets["audio"],
        targets["visual"],
    )

    return targets
