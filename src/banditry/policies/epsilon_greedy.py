import numpy as np

from .base import BasePolicy


class EpsilonGreedyPolicy(BasePolicy):
    def __init__(self, bandit, epsilon: float = 0.1):
        super().__init__(bandit)
        self._epsilon = float(epsilon)  # Probability to explore
        self._update_method = lambda n: 1 / n  # Updating method (average sampling by default)

        self._initial_value_estimates = np.zeros(self._bandit.n_arms)
        self._value_estimates = self._initial_value_estimates.copy()

    def _select_action(self):
        is_greedy = np.random.uniform() > self._epsilon
        return self._exploitation() if is_greedy else self._exploration()

    def _exploitation(self):
        # Break ties randomly among max actions
        candidates = self._find_best_arms(self._value_estimates)
        return self._break_ties(candidates)

    def _exploration(self):
        return int(np.random.randint(0, self._bandit.n_arms))

    def _update(self, action, reward):
        self._action_counts[action] = self._action_counts[action] + 1
        self._value_estimates[action] += self._update_method(self._action_counts[action]) * (
            reward - self._value_estimates[action]
        )

    def reset(self):
        self._step = 0
        self._value_estimates = self._initial_value_estimates.copy()
        self._action_counts = np.zeros(self._bandit.n_arms)
        self._action_history.clear()
        self._reward_history.clear()

    def set_initial_values(self, initial_values):
        if len(initial_values) != self._bandit.n_arms:
            raise ValueError(f"Length mismatch: Q({len(initial_values)}) must be of size {self._bandit.n_arms}.")
        self._initial_value_estimates = np.array(initial_values, dtype=float)
        if self._step == 0:
            self._value_estimates = self._initial_value_estimates.copy()

    def set_epsilon(self, epsilon):
        if epsilon < 0 or epsilon > 1:
            raise ValueError("Value must be between 0 and 1 (inclusive)")
        self._epsilon = float(epsilon)

    def set_update_method(self, method):
        self.reset()
        self._update_method = method
