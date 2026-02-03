import numpy as np

from .base import BasePolicy


class UCBPolicy(BasePolicy):
    def __init__(self, bandit, delta: float):
        super().__init__(bandit)
        if delta <= 0 or delta >= 1:
            raise ValueError("delta must be in (0, 1).")
        self._delta = float(delta)
        self._bonus = np.full(self._bandit.n_arms, np.inf)

    def _select_action(self):
        ucb = self.value_estimates + self._bonus
        candidates = self._find_best_arms(ucb)
        return self._break_ties(candidates)

    def _update(self, action, reward):
        super()._update(action, reward)
        count = self.action_counts[action]
        self._bonus[action] = np.sqrt(
            2 * np.log(1 / self._delta) / count
        )
