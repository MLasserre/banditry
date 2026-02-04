from abc import ABC, abstractmethod
from typing import Dict, Optional, Type

import numpy as np

from .._types import InitialEstimates
from ..estimators import BaseEstimator, SampleMeanEstimator


class BasePolicy(ABC):
    def __init__(
        self,
        bandit,
        initial_estimates: InitialEstimates = None,
        *,
        estimator_cls: Type[BaseEstimator] = SampleMeanEstimator,
        estimator_kwargs: Optional[Dict[str, object]] = None,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ):
        self._bandit = bandit
        self._step = 0
        self._rng = self._build_rng(rng=rng, seed=seed)

        self._estimator = self._build_estimator(
            initial_estimates=initial_estimates,
            estimator_cls=estimator_cls,
            estimator_kwargs=estimator_kwargs,
        )

        self._action_history = []
        self._reward_history = []

    def _build_rng(
        self,
        rng: Optional[np.random.Generator],
        seed: Optional[int],
    ) -> np.random.Generator:
        if rng is not None and seed is not None:
            raise ValueError("Provide either rng or seed, not both.")
        if rng is not None:
            if not isinstance(rng, np.random.Generator):
                raise TypeError("rng must be a numpy.random.Generator instance.")
            return rng
        return np.random.default_rng(seed)

    def _build_estimator(
        self,
        initial_estimates: InitialEstimates,
        estimator_cls: Type[BaseEstimator],
        estimator_kwargs: Optional[Dict[str, object]],
    ) -> BaseEstimator:
        if not isinstance(estimator_cls, type):
            raise TypeError("estimator_cls must be an estimator class.")
        if not issubclass(estimator_cls, BaseEstimator):
            raise TypeError("estimator_cls must inherit from BaseEstimator.")

        kwargs = {} if estimator_kwargs is None else dict(estimator_kwargs)
        reserved_keys = {"size", "initial_estimates"}
        overlapping_keys = reserved_keys.intersection(kwargs)
        if overlapping_keys:
            keys = ", ".join(sorted(overlapping_keys))
            raise ValueError(
                f"estimator_kwargs cannot contain reserved key(s): {keys}."
            )

        estimator = estimator_cls(
            size=self._bandit.n_arms,
            initial_estimates=initial_estimates,
            **kwargs,
        )

        if estimator.size != self._bandit.n_arms:
            raise ValueError(
                "Estimator size mismatch: estimator size "
                f"{estimator.size} != number of arms {self._bandit.n_arms}."
            )
        return estimator

    def _break_ties(self, candidates):
        """Break ties if multiple arms are optimal."""
        if len(candidates) == 1:
            return int(candidates[0])
        return int(self._rng.choice(candidates))

    def _find_best_arms(self, values):
        return np.flatnonzero(values == np.max(values))

    def _record(self, action, reward):
        self._action_history.append(action)
        self._reward_history.append(reward)

    def _update(self, action, reward):
        self._estimator.update(action, reward)

    @abstractmethod
    def _select_action(self):
        pass

    @property
    def step(self):
        return self._step

    @property
    def estimator(self) -> BaseEstimator:
        return self._estimator

    @property
    def value_estimates(self) -> np.ndarray:
        return self._estimator.estimates

    @property
    def action_counts(self) -> np.ndarray:
        return self._estimator.counts

    def value_estimate_at(self, action: int) -> float:
        return self._estimator.estimate_at(action)

    def action_count_at(self, action: int) -> int:
        return self._estimator.count_at(action)

    @property
    def action_history(self):
        return self._action_history

    @property
    def reward_history(self):
        return self._reward_history

    @property
    def history(self):
        return self.action_history, self.reward_history

    def learn(self, n_step: int):
        if not isinstance(n_step, int):
            raise TypeError("n_step must be an int.")
        if n_step < 0:
            raise ValueError("n_step must be non-negative.")

        for _ in range(n_step):
            action = self._select_action()
            reward, _ = self._bandit.pull(action)
            self._update(action, reward)
            self._record(action, reward)
            self._step += 1

        return self._action_history[-n_step:], self._reward_history[-n_step:]
