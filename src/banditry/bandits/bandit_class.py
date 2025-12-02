from typing import List, Optional, Sequence
import numpy as np

from .._types import Action, Sample
from .arm_class import BaseArm


class Bandit:
    """A multi-armed bandit that delegates sampling to its arms."""

    def __init__(self, arms: Sequence[BaseArm], seed: Optional[int] = None):
        if not arms:
            raise ValueError("At least one arm is required.")
        self._arms: List[BaseArm] = list(arms)
        self._rng = np.random.default_rng(seed)
        self._step = 0

    @property
    def n_arms(self) -> int:
        return len(self._arms)

    @property
    def step(self) -> int:
        return self._step

    def pull(self, action: Action) -> Sample:
        """Sample reward/info from the chosen arm."""
        if action < 0 or action >= self.n_arms:
            raise ValueError(f"Invalid arm {action} (expected 0–{self.n_arms - 1})")
        reward, info = self._arms[action].sample(self._rng)
        self._step += 1
        return reward, info
