
from pathlib import Path
import numpy as np


def load_video_features(
    video_id: str,
    features_dir: str | Path,
) -> dict[str, np.ndarray]:

    features_dir = Path(features_dir)

    audio = np.load(
        features_dir / "vggish" / f"{video_id}.npy",
        allow_pickle=False,
    )

    visual = np.load(
        features_dir / "res152" / f"{video_id}.npy",
        allow_pickle=False,
    )

    motion = np.load(
        features_dir / "r2plus1d_18" / f"{video_id}.npy",
        allow_pickle=False,
    )

    if audio.shape != (10, 128):
        raise ValueError("Unexpected audio feature shape.")

    if visual.shape != (80, 2048):
        raise ValueError("Unexpected visual feature shape.")

    if motion.shape != (10, 512):
        raise ValueError("Unexpected motion feature shape.")

    # Average 8 visual frames into one temporal segment
    visual = visual.reshape(10, 8, 2048).mean(axis=1)

    return {
        "audio": audio.astype(np.float32),
        "visual": visual.astype(np.float32),
        "motion": motion.astype(np.float32),
    }
