
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess

import numpy as np

from eventlens.frame_extraction import (
    extract_frames,
    select_uniform_frames,
)
from eventlens.rgb_features import extract_rgb_features
from eventlens.motion_features import extract_motion_features
from eventlens.audio_features import extract_audio_features
from eventlens.vggish_setup import prepare_vggish_extractor


def extract_video_features(
    video_path: Path,
    rgb_model,
    motion_model,
    device,
    vggish_archive: Path,
    vggish_dir: Path,
) -> dict:

    video_path = Path(video_path)

    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    # Our current models work on 10-second clips
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    duration = float(probe.stdout.strip())

    if not 9.5 <= duration <= 10.5:
        raise ValueError(
            "Video must be approximately 10 seconds long"
        )

    prepare_vggish_extractor(
        vggish_archive,
        vggish_dir,
    )

    with TemporaryDirectory() as temp:
        temp_dir = Path(temp)

        # 1. Extract video frames
        frames = extract_frames(
            video_path,
            temp_dir / "frames",
            fps=8,
        )

        selected = select_uniform_frames(
            frames,
            count=80,
        )

        # 2. Extract RGB features
        rgb = extract_rgb_features(
            selected,
            rgb_model,
            device,
            batch_size=8,
        )

        # Average each group of 8 frames
        visual = rgb.reshape(
            10, 8, 2048
        ).mean(axis=1)

        # 3. Extract motion features
        motion = extract_motion_features(
            selected,
            motion_model,
            device,
            batch_size=1,
        )

        # 4. Extract 10-second WAV audio
        wav_path = temp_dir / "audio.wav"

        subprocess.run(
            [
                "ffmpeg", "-nostdin",
                "-hide_banner", "-loglevel", "error",
                "-i", str(video_path),
                "-vn",
                "-af", "apad=whole_dur=10",
                "-t", "10",
                "-ac", "1",
                "-ar", "16000",
                "-c:a", "pcm_s16le",
                "-y", str(wav_path),
            ],
            check=True,
        )

        # 5. Extract VGGish audio features
        audio = extract_audio_features(
            wav_path=wav_path,
            checkpoint_path=(
                Path(vggish_dir) / "vggish_model.ckpt"
            ),
            extractor_dir=vggish_dir,
        )

    features = {
        "audio": audio.astype(np.float32),
        "visual": visual.astype(np.float32),
        "motion": motion.astype(np.float32),
    }

    expected = {
        "audio": (10, 128),
        "visual": (10, 2048),
        "motion": (10, 512),
    }

    for name, array in features.items():
        if array.shape != expected[name]:
            raise ValueError(
                f"{name}: unexpected shape {array.shape}"
            )

        if not np.isfinite(array).all():
            raise ValueError(
                f"{name}: non-finite values"
            )

    return features
