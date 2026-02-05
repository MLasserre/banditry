from dataclasses import dataclass

from .base import BaseSchedule


@dataclass(frozen=True)
class ConstantSchedule(BaseSchedule):
    """Keep a parameter constant over time."""

    initial_value: float

    def value(self, step: int) -> float:
        if step < 0:
            raise ValueError("step must be non-negative.")
        return float(self.initial_value)
