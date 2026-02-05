from abc import ABC, abstractmethod


class BaseSchedule(ABC):
    """Base interface for scalar schedules driven by step index."""

    @abstractmethod
    def value(self, step: int, start: float) -> float:
        """Return scheduled value at a given step."""
        pass

    def __call__(self, step: int, start: float) -> float:
        return self.value(step=step, start=start)
