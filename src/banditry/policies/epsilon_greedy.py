from typing import Dict, Optional, Sequence, Type, Union

import numpy as np

from ..estimators import BaseEstimator, EWMeanEstimator, SampleMeanEstimator
from .base import BasePolicy


class EpsilonGreedyPolicy(BasePolicy):
    _ESTIMATOR_ALIASES = {
        "sample_mean": SampleMeanEstimator,
        "ew_mean": EWMeanEstimator,
    }

    def __init__(
        self,
        bandit,
        epsilon: float = 0.1,
        initial_estimates: Optional[Sequence[float]] = None,
        *,
        estimator: Union[str, Type[BaseEstimator]] = "sample_mean",
        estimator_kwargs: Optional[Dict[str, object]] = None,
    ):
        estimator_cls = self._resolve_estimator_class(estimator)
        super().__init__(
            bandit,
            initial_estimates=initial_estimates,
            estimator_cls=estimator_cls,
            estimator_kwargs=estimator_kwargs,
        )
        self.set_epsilon(epsilon)

    @classmethod
    def _resolve_estimator_class(
        cls, estimator: Union[str, Type[BaseEstimator]]
    ) -> Type[BaseEstimator]:
        if isinstance(estimator, str):
            normalized = estimator.strip().lower()
            if normalized not in cls._ESTIMATOR_ALIASES:
                supported = ", ".join(sorted(cls._ESTIMATOR_ALIASES))
                raise ValueError(
                    f"Unsupported estimator alias '{estimator}'. "
                    f"Expected one of: {supported}."
                )
            return cls._ESTIMATOR_ALIASES[normalized]

        if not isinstance(estimator, type):
            raise TypeError(
                "estimator must be a string alias or an estimator class."
            )
        if not issubclass(estimator, BaseEstimator):
            raise TypeError("estimator class must inherit from BaseEstimator.")
        return estimator

    def _select_action(self):
        is_greedy = np.random.uniform() > self._epsilon
        return self._exploitation() if is_greedy else self._exploration()

    def _exploitation(self):
        # Break ties randomly among max actions
        candidates = self._find_best_arms(self.value_estimates)
        return self._break_ties(candidates)

    def _exploration(self):
        return int(np.random.randint(0, self._bandit.n_arms))

    def set_epsilon(self, epsilon):
        if epsilon < 0 or epsilon > 1:
            raise ValueError("Value must be between 0 and 1 (inclusive)")
        self._epsilon = float(epsilon)
