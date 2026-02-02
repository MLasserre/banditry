from typing import Optional, Sequence

from .base import BaseEstimator


class SampleMeanEstimator(BaseEstimator):
    """Online estimator using the incremental sample average."""

    def __init__(self, size: int, initial_estimates: Optional[Sequence[float]] = None):
        super().__init__(size=size, initial_estimates=initial_estimates)

    def update(self, index: int, sample: float) -> None:
        self._counts[index] += 1
        self._estimates[index] += (sample - self._estimates[index]) / self._counts[index]