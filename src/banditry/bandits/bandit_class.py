from typing import Dict, List, Optional, Sequence
import numpy as np

from .._types import Action, Sample
from .arm_class import BaseArm


class Bandit:
    """A multi-armed bandit that delegates sampling to its arms."""

    def __init__(self, arms: Sequence[BaseArm], seed: Optional[int] = None):
        if not arms:
            raise ValueError("At least one arm is required.")
        self._arms: List[BaseArm] = list(arms)
        # Build per-bandit labels without mutating arms; auto-name unnamed, reject duplicates
        self._labels: List[str] = []
        self._name_to_index: Dict[str, int] = {}
        for idx, arm in enumerate(self._arms):
            label = arm.name or f"arm{idx}"
            if label in self._name_to_index:
                raise ValueError(f"Duplicate arm label '{label}' at indices {self._name_to_index[label]} and {idx}")
            self._labels.append(label)
            self._name_to_index[label] = idx
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
        arm_index = self._resolve_action(action)
        reward, info = self._arms[arm_index].sample(self._rng)
        # Structural metadata always set by the bandit
        info["arm_index"] = arm_index
        info["arm_name"] = self._labels[arm_index]
        self._step += 1
        return reward, info

    def _resolve_action(self, action: Action) -> int:
        if isinstance(action, str):
            if action not in self._name_to_index:
                raise ValueError(f"Unknown arm name '{action}'")
            return self._name_to_index[action]
        # action is expected to be an int
        if action < 0 or action >= self.n_arms:
            raise ValueError(f"Invalid arm {action} (expected 0–{self.n_arms - 1})")
        return int(action)

    # Bandits are immutable after construction: no add/remove
