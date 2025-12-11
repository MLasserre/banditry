from typing import Dict, List, Optional, Sequence
import numpy as np

from .._types import Action, Sample
from .arms import BaseArm


class Bandit:
    """
    A multi-armed bandit that delegates sampling to its arms.

    Set restless=True to evolve every arm each bandit step (restless case); by
    default only the pulled arm advances (rested case).
    """

    def __init__(
        self,
        arms: Sequence[BaseArm],
        seed: Optional[int] = None,
        restless: bool = False,
    ):
        if not arms:
            raise ValueError("At least one arm is required.")
        self._arms: List[BaseArm] = list(arms)
        # Build per-bandit labels without mutating arms; auto-name unnamed, reject duplicates
        self._arm_labels: List[str] = []
        self._label_to_index: Dict[str, int] = {}
        for idx, arm in enumerate(self._arms):
            label = arm.name or f"arm{idx}"
            if label in self._label_to_index:
                raise ValueError(f"Duplicate arm label '{label}' at indices {self._label_to_index[label]} and {idx}")
            self._arm_labels.append(label)
            self._label_to_index[label] = idx
        self._rng = np.random.default_rng(seed)
        self._num_pulls = 0
        self._restless = restless
        self._step = 0

    @property
    def n_arms(self) -> int:
        return len(self._arms)

    @property
    def num_pulls(self) -> int:
        return self._num_pulls

    def arm_num_pulls(self, action: Action) -> int:
        """Return the number of pulls for a given arm (by index or label)."""
        arm_index = self._resolve_action(action)
        return self._arms[arm_index].num_pulls

    def pull(self, action: Action) -> Sample:
        """Sample reward/info from the chosen arm."""
        arm_index = self._resolve_action(action)
        reward, info = self._arms[arm_index].sample(self._rng)
        # Structural metadata always set by the bandit
        info["arm_index"] = arm_index
        info["arm_label"] = self._arm_labels[arm_index]
        # Track pulls per arm
        self._arms[arm_index]._record_pull()
        self._num_pulls += 1
        # Advance global time after sampling; evolution applies to future pulls
        self._step += 1
        if self._restless:
            for arm in self._arms:
                if not arm.is_stationary:
                    arm.evolve(self._rng, self._step)
        else:
            arm = self._arms[arm_index]
            if not arm.is_stationary:
                arm.evolve(self._rng, self._step)
        return reward, info

    def __repr__(self) -> str:
        labels = ", ".join(self._arm_labels)
        return f"Bandit(n_arms={self.n_arms}, num_pulls={self._num_pulls}, labels=[{labels}])"

    def _resolve_action(self, action: Action) -> int:
        if isinstance(action, str):
            if action not in self._label_to_index:
                raise ValueError(f"Unknown arm name '{action}'")
            return self._label_to_index[action]
        # action is expected to be an int
        if action < 0 or action >= self.n_arms:
            raise ValueError(f"Invalid arm {action} (expected 0–{self.n_arms - 1})")
        return int(action)

    # Bandits are immutable after construction: no add/remove
