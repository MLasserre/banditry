from dataclasses import dataclass

from .base import BaseSchedule


@dataclass(frozen=True)
class LinearDecaySchedule(BaseSchedule):
    """Linearly interpolate from initial_value toward min_value over decay_steps."""

    initial_value: float
    min_value: float = 0.0
    decay_steps: int = 1000

    def __post_init__(self):
        if self.decay_steps <= 0:
            raise ValueError("decay_steps must be strictly positive.")

    def value(self, step: int) -> float:
        if step < 0:
            raise ValueError("step must be non-negative.")
        progress = min(1.0, int(step) / self.decay_steps)
        return float(
            self.initial_value + progress * (self.min_value - self.initial_value)
        )
