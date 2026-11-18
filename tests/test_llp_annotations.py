
import pytest

from eventlens.llp_annotations import load_temporal_annotations


@pytest.mark.parametrize("modality", ["audio", "visual"])
def test_validation_only(tmp_path, modality):
    (tmp_path / "AVVP_val_pd.csv").write_text(
        "filename\tevent_labels\n"
        "val_video\tClapping\n",
        encoding="utf-8",
    )

    (tmp_path / f"AVVP_eval_{modality}.csv").write_text(
        "filename\tonset\toffset\tevent_labels\n"
        "val_video\t0\t4\tClapping\n"
        "test_video\t2\t6\tDog\n"
        "unknown_video\t1\t5\tSpeech\n",
        encoding="utf-8",
    )

    events = load_temporal_annotations(
        "validation", modality, tmp_path
    )

    assert len(events) == 1
    assert events[0]["filename"] == "val_video"
    assert events[0]["label"] == "Clapping"
    assert events[0]["start_sec"] == 0
    assert events[0]["end_sec"] == 4
    assert events[0]["modality"] == modality


def test_invalid_split(tmp_path):
    with pytest.raises(ValueError):
        load_temporal_annotations("training", "audio", tmp_path)


def test_invalid_modality(tmp_path):
    with pytest.raises(ValueError):
        load_temporal_annotations("validation", "sound", tmp_path)
