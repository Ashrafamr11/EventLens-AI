
import pytest

from eventlens.audio_extraction import extract_audio


def test_missing_video(tmp_path):
    video = tmp_path / "missing.mp4"
    output = tmp_path / "audio.wav"

    with pytest.raises(FileNotFoundError):
        extract_audio(video, output)


def test_invalid_output_format(tmp_path):
    video = tmp_path / "sample.mp4"
    video.touch()

    with pytest.raises(ValueError):
        extract_audio(video, tmp_path / "audio.mp3")


def test_existing_audio_file(tmp_path):
    video = tmp_path / "sample.mp4"
    video.touch()

    output = tmp_path / "audio.wav"
    output.touch()

    with pytest.raises(FileExistsError):
        extract_audio(video, output)
