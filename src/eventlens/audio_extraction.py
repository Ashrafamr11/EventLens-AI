
import subprocess
from pathlib import Path


def extract_audio(
    video_path: str | Path,
    output_path: str | Path,
) -> Path:
    video_path = Path(video_path)
    output_path = Path(output_path)

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    if output_path.suffix.lower() != ".wav":
        raise ValueError("Output must be a WAV file.")

    if output_path.exists():
        raise FileExistsError(
            f"Audio file already exists: {output_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-n",
        "-i", str(video_path),
        "-map", "0:a:0",
        "-vn",
        "-c:a", "pcm_s16le",
        str(output_path),
    ]

    subprocess.run(command, check=True)

    return output_path
