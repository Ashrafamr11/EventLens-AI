
import numpy as np


def build_class_names(train_samples: list[dict]) -> list[str]:
    classes = sorted({
        label
        for sample in train_samples
        for label in sample["labels"]
    })

    if not classes:
        raise ValueError("No event classes found.")

    return classes


def encode_multilabel(
    labels: list[str],
    class_names: list[str],
) -> np.ndarray:

    class_to_index = {
        name: index
        for index, name in enumerate(class_names)
    }

    encoded = np.zeros(len(class_names), dtype=np.float32)

    for label in labels:
        if label not in class_to_index:
            raise ValueError(f"Unknown event label: {label}")

        encoded[class_to_index[label]] = 1.0

    return encoded
