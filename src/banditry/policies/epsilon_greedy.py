from typing import Dict, Optional, Type, Union

import numpy as np

from .._types import InitialEstimates
from ..estimators import BaseEstimator, EWMeanEstimator, SampleMeanEstimator
from ..schedules import BaseSchedule, ConstantSchedule
from .base import BasePolicy


class EpsilonGreedyPolicy(BasePolicy):
    _ESTIMATOR_ALIASES = {
        "sample_mean": SampleMeanEstimator,
        "ew_mean": EWMeanEstimator,
    }

    def __init__(
        self,
        bandit,
        epsilon: Union[float, BaseSchedule] = 0.1,
        initial_estimates: InitialEstimates = None,
        *,
        estimator: Union[str, Type[BaseEstimator]] = "sample_mean",
        estimator_kwargs: Optional[Dict[str, object]] = None,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ):
        estimator_cls = self._resolve_estimator_class(estimator)
        super().__init__(
            bandit,
            initial_estimates=initial_estimates,
            estimator_cls=estimator_cls,
            estimator_kwargs=estimator_kwargs,
            rng=rng,
            seed=seed,
        )
        self._epsilon_schedule = self._resolve_epsilon_schedule(epsilon=epsilon)
        self._epsilon = self._scheduled_epsilon(step=0)

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

    @staticmethod
    def _resolve_epsilon_schedule(
        epsilon: Union[float, BaseSchedule],
    ) -> BaseSchedule:
        if isinstance(epsilon, BaseSchedule):
            return epsilon
        try:
            epsilon_value = float(epsilon)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "epsilon must be a float or an instance of BaseSchedule."
            ) from exc
        return ConstantSchedule(
            initial_value=EpsilonGreedyPolicy._validate_epsilon(epsilon_value)
        )

    @staticmethod
    def _validate_epsilon(epsilon: float) -> float:
        if epsilon < 0 or epsilon > 1:
            raise ValueError("epsilon must be between 0 and 1 (inclusive).")
        return float(epsilon)

    def _scheduled_epsilon(self, step: int) -> float:
        epsilon = self._epsilon_schedule.value(step=step)
        return self._validate_epsilon(epsilon)

    def _select_action(self):
        is_greedy = self._rng.random() > self._epsilon
        return self._exploitation() if is_greedy else self._exploration()

    def _exploitation(self):
        # Break ties randomly among max actions
        candidates = self._find_best_arms(self.value_estimates)
        return self._break_ties(candidates)

    def _exploration(self):
        return int(self._rng.integers(0, self._bandit.n_arms))

    def _update(self, action, reward):
        super()._update(action, reward)
        self._epsilon = self._scheduled_epsilon(self._step + 1)

    @property
    def epsilon(self) -> float:
        return self._epsilon
