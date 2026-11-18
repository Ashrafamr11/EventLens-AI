
import numpy as np


def decode_segment_events(
    segment_probabilities,
    class_names,
    threshold=0.1,
):
    probs = np.asarray(
        segment_probabilities,
        dtype=np.float32,
    )

    if probs.ndim != 2:
        raise ValueError("Expected (segments, classes).")

    num_segments, num_classes = probs.shape

    if num_classes != len(class_names):
        raise ValueError("Class count mismatch.")

    if not 0 <= threshold <= 1:
        raise ValueError("Invalid threshold.")

    if not np.isfinite(probs).all():
        raise ValueError("Non-finite probabilities.")

    events = []

    for class_idx, label in enumerate(class_names):
        start = None

        for t in range(num_segments + 1):
            active = (
                t < num_segments
                and probs[t, class_idx] >= threshold
            )

            if active and start is None:
                start = t

            elif not active and start is not None:
                confidence = float(
                    probs[start:t, class_idx].mean()
                )

                events.append({
                    "label": label,
                    "start_sec": float(start),
                    "end_sec": float(t),
                    "confidence": confidence,
                })

                start = None

    return sorted(
        events,
        key=lambda event: (
            event["start_sec"],
            -event["confidence"],
        ),
    )
