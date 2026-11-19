
import numpy as np
import torch


def predict_events(
    model,
    features: dict,
    class_names: list[str],
    threshold: float = 0.30,
) -> list[dict]:

    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")

    expected = {
        "audio": (10, 128),
        "visual": (10, 2048),
        "motion": (10, 512),
    }

    inputs = {}

    for name, shape in expected.items():
        value = np.asarray(features[name], dtype=np.float32)

        if value.shape != shape:
            raise ValueError(
                f"{name}: expected {shape}, got {value.shape}"
            )

        if not np.isfinite(value).all():
            raise ValueError(f"Invalid {name} features")

        inputs[name] = torch.from_numpy(
            value.copy()
        ).unsqueeze(0)

    model.eval()

    with torch.inference_mode():
        outputs = model(
            inputs["audio"],
            inputs["visual"],
            inputs["motion"],
        )

        probabilities = torch.sigmoid(
            outputs["segment_logits"][0]
        ).cpu().numpy()

    if probabilities.shape != (10, len(class_names)):
        raise ValueError("Unexpected prediction shape")

    events = []

    for class_index, label in enumerate(class_names):
        start = None

        for second in range(11):
            active = (
                second < 10
                and probabilities[second, class_index] >= threshold
            )

            if active and start is None:
                start = second

            elif not active and start is not None:
                confidence = float(
                    probabilities[
                        start:second, class_index
                    ].mean()
                )

                events.append({
                    "label": label,
                    "start_sec": start,
                    "end_sec": second,
                    "confidence": confidence,
                })

                start = None

    events.sort(
        key=lambda event: (
            event["start_sec"],
            -event["confidence"],
        )
    )

    return events
