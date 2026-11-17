
from dataclasses import dataclass


@dataclass
class DetectedEvent:
    label: str
    start_sec: float
    end_sec: float
    confidence: float

    def __post_init__(self):
        if not self.label.strip():
            raise ValueError("Event label cannot be empty.")

        if self.start_sec < 0 or self.end_sec <= self.start_sec:
            raise ValueError("Invalid event timestamps.")

        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1.")
