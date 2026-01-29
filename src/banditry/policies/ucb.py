import numpy as np

from .base import BasePolicy


class UCBPolicy(BasePolicy):
    def __init__(self, bandit, delta: int):
        super().__init__(bandit)
        self._delta = delta
        self._bonus = np.full(self._bandit.n_arms, np.inf)

    def _select_action(self):
        ucb = self._value_estimates + self._bonus
        candidates = self._find_best_arms(ucb)
        return self._break_ties(candidates)

    def _update(self, action, reward):
        self._action_counts[action] += 1
        self._value_estimates[action] += (
            reward - self._value_estimates[action]
        ) / self._action_counts[action]
        self._bonus[action] = np.sqrt(
            2 * np.log(1 / self._delta) / self._action_counts[action]
        )
