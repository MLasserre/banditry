from .base import BaseEstimator, InitialEstimates


class EWMeanEstimator(BaseEstimator):
    """Exponential recency-weighted mean estimator with constant alpha."""

    def __init__(
        self,
        size: int,
        alpha: float = 0.1,
        initial_estimates: InitialEstimates = None,
    ):
        super().__init__(size=size, initial_estimates=initial_estimates)
        if alpha <= 0 or alpha > 1:
            raise ValueError("alpha must be in (0, 1].")
        self._alpha = float(alpha)

    @property
    def alpha(self) -> float:
        return self._alpha

    def update(self, index: int, sample: float) -> None:
        self._counts[index] += 1
        self._estimates[index] += self._alpha * (sample - self._estimates[index])
