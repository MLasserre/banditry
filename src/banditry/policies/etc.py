from .base import BasePolicy


class ETCPolicy(BasePolicy):
    def __init__(self, bandit, m: int):
        super().__init__(bandit)
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
