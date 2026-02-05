from dataclasses import dataclass

from .base import BaseSchedule


@dataclass(frozen=True)
class LinearDecaySchedule(BaseSchedule):
    """Linearly interpolate from start toward min_value over decay_steps."""

    min_value: float = 0.0
    decay_steps: int = 1000

    def __post_init__(self):
        if self.decay_steps <= 0:
            raise ValueError("decay_steps must be strictly positive.")

    def value(self, step: int, start: float) -> float:
        if step < 0:
            raise ValueError("step must be non-negative.")
        progress = min(1.0, int(step) / self.decay_steps)
        return float(start + progress * (self.min_value - start))
