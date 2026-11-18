
import subprocess
from pathlib import Path


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    fps: float = 1.0,
) -> list[Path]:

    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    if fps <= 0:
        raise ValueError("FPS must be greater than zero.")

    output_dir.mkdir(parents=True, exist_ok=True)

    if list(output_dir.glob("frame_*.jpg")):
        raise FileExistsError(
            "Output directory already contains extracted frames."
        )

    output_pattern = output_dir / "frame_%03d.jpg"

    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "3",
        str(output_pattern),
    ]

    subprocess.run(command, check=True)

    return sorted(output_dir.glob("frame_*.jpg"))
