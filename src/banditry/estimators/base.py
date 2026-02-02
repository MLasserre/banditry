from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np


class BaseEstimator(ABC):
    """Base interface for online estimators over indexed streams."""

    def __init__(self, size: int, initial_estimates: Optional[Sequence[float]] = None):
        if size <= 0:
            raise ValueError("size must be strictly positive.")
        self._size = int(size)
        self._counts = np.zeros(self._size, dtype=int)
        if initial_estimates is None:
            self._estimates = np.zeros(self._size)
        else:
            if len(initial_estimates) != self._size:
                raise ValueError(
                    f"Length mismatch: initial_estimates({len(initial_estimates)}) must be of size {self._size}."
                )
            self._estimates = np.array(initial_estimates, dtype=float)

    @property
    def size(self) -> int:
        return self._size

    @property
    def estimates(self) -> np.ndarray:
        return self._estimates

    def estimate_at(self, index: int) -> float:
        return float(self._estimates[index])

    @property
    def counts(self) -> np.ndarray:
        return self._counts

    def count_at(self, index: int) -> int:
        return int(self._counts[index])

    @abstractmethod
    def update(self, index: int, sample: float) -> None:
        """Update estimator state after observing one sample at an index."""
        pass
