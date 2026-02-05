from dataclasses import dataclass

from .base import BaseSchedule


@dataclass(frozen=True)
class ExponentialDecaySchedule(BaseSchedule):
    """Exponentially decay from start with optional floor min_value."""

    min_value: float = 0.0
    decay: float = 0.99

    def __post_init__(self):
        if self.decay <= 0 or self.decay > 1:
            raise ValueError("decay must be in (0, 1].")

    def value(self, step: int, start: float) -> float:
        if step < 0:
            raise ValueError("step must be non-negative.")
        return float(max(self.min_value, start * (self.decay ** int(step))))
