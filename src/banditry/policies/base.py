from abc import ABC, abstractmethod

import numpy as np


class BasePolicy(ABC):
    def __init__(self, bandit):
        self._bandit = bandit
        self._step = 0

        self._value_estimates = np.zeros(bandit.n_arms)
        self._action_counts = np.zeros(bandit.n_arms)

        self._action_history = []
        self._reward_history = []

    def _break_ties(self, candidates):
        """Break ties if multiple arms are optimal."""
        if len(candidates) == 1:
            return int(candidates[0])
        return int(np.random.choice(candidates))

    def _find_best_arms(self, values):
        return np.flatnonzero(values == np.max(values))

    def _record(self, action, reward):
        self._action_history.append(action)
        self._reward_history.append(reward)

    @abstractmethod
    def _update(self, action, reward):
        pass

    def _incremental_update(self, action, reward):
        self._action_counts[action] += 1
        self._value_estimates[action] += (
            reward - self._value_estimates[action]
        ) / self._action_counts[action]

    @abstractmethod
    def _select_action(self):
        pass

    @property
    def step(self):
        return self._step

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
