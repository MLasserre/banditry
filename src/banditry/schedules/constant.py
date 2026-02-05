from .base import BaseSchedule


class ConstantSchedule(BaseSchedule):
    """Keep a parameter constant over time."""

    def value(self, step: int, start: float) -> float:
        return float(start)
