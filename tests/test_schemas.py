
import pytest

from eventlens.schemas import DetectedEvent


def test_valid_event():
    event = DetectedEvent(
        label="Dog Barking",
        start_sec=12.0,
        end_sec=18.0,
        confidence=0.95,
    )

    assert event.label == "Dog Barking"
    assert event.start_sec == 12.0
    assert event.end_sec == 18.0
    assert event.confidence == 0.95


def test_invalid_timestamps():
    with pytest.raises(ValueError):
        DetectedEvent(
            label="Dog Barking",
            start_sec=20.0,
            end_sec=10.0,
            confidence=0.95,
        )
