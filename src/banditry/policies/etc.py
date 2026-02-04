from typing import Optional

import numpy as np

from .base import BasePolicy


class ETCPolicy(BasePolicy):
    def __init__(
        self,
        bandit,
        m: int,
        *,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(bandit, rng=rng, seed=seed)
        self._m = m
        self._best_action = None

    def _select_action(self):
        if self._step < self._m * self._bandit.n_arms:
            return self._exploration()
        return self._exploitation()

    def _exploration(self):
        return self._step % self._bandit.n_arms

    def _exploitation(self):
        if self._best_action is None:
            candidates = self._find_best_arms(self.value_estimates)
            self._best_action = self._break_ties(candidates)
        return self._best_action
