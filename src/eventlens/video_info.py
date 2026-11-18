
import json
import subprocess
from pathlib import Path
from fractions import Fraction


def probe_video(video_path: str | Path) -> dict:
    video_path = Path(video_path)

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    command = [
        "ffprobe",
        "-v", "error",
        "-show_entries",
        "format=duration:stream=codec_type,codec_name,width,height,avg_frame_rate,sample_rate,channels",
        "-of", "json",
        str(video_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    return json.loads(result.stdout)

def get_video_metadata(video_path: str | Path) -> dict:
    raw_data = probe_video(video_path)
    streams = raw_data["streams"]

    video = next(
        (s for s in streams if s["codec_type"] == "video"),
        None,
    )

    audio = next(
        (s for s in streams if s["codec_type"] == "audio"),
        None,
    )

    if video is None:
        raise ValueError("No video stream found.")

    duration = float(raw_data["format"]["duration"])
    fps = float(Fraction(video["avg_frame_rate"]))

    if duration <= 0 or fps <= 0:
        raise ValueError("Invalid video duration or FPS.")

    return {
        "duration_sec": duration,
        "width": video["width"],
        "height": video["height"],
        "fps": fps,
        "has_audio": audio is not None,
        "sample_rate": (
            int(audio["sample_rate"])
            if audio and "sample_rate" in audio
            else None
        ),
        "channels": (
            audio.get("channels")
            if audio
            else None
        ),
    }
