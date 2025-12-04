from abc import ABC, abstractmethod
from typing import Callable, Optional, Union, Tuple
import numpy as np
from .._types import Reward, Sample


class BaseArm(ABC):
    def __init__(self, name: Optional[str] = None):
        # Name can be supplied by the user; otherwise stays None and the bandit will assign a label.
        self._name = name

    @property
    def name(self) -> Optional[str]:
        return self._name

    @abstractmethod
    def expected_reward(self) -> Reward:
        """Return the mean reward of this arm."""

    @abstractmethod
    def sample(self, rng: np.random.Generator) -> Sample:
        """Draw a reward from this arm."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self._name!r}, expected_reward={self.expected_reward():.4g})"


class BernoulliArm(BaseArm):
    def __init__(self, p: float, name: Optional[str] = None):
        super().__init__(name)
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0, 1].")
        self._p = float(p)

    def sample(self, rng: np.random.Generator) -> Sample:
        r = float(rng.binomial(1, self._p))
        return r, {"type": "bernoulli", "p": self._p, "name": self._name}

    def expected_reward(self) -> Reward:
        return self._p

    def __repr__(self) -> str:
        return f"BernoulliArm(p={self._p}, name={self._name!r})"


class GaussianArm(BaseArm):
    def __init__(self, mean: float, std: float, name: Optional[str] = None):
        super().__init__(name)
        if std <= 0:
            raise ValueError("std must be positive.")
        self._mu = float(mean)
        self._sigma = float(std)

    def sample(self, rng: np.random.Generator):
        reward = rng.normal(self._mu, self._sigma)
        return reward, {"type": "Gaussian", "mean": self._mu, "std": self._sigma, "name": self._name}

    def expected_reward(self) -> Reward:
        return self._mu

    def __repr__(self) -> str:
        return f"GaussianArm(mean={self._mu}, std={self._sigma}, name={self._name!r})"


class CustomArm(BaseArm):
    """
    Arm backed by a user-provided sampling function.

    sample_fn must accept an rng and return either:
      - a reward (float)
      - or a tuple (reward, info_dict)
    """

    def __init__(
        self,
        sample_fn: Callable[[np.random.Generator], Union[Reward, Tuple[Reward, dict]]],
        expected_reward_value: Optional[float] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self._sample_fn = sample_fn
        self._expected_reward_value = expected_reward_value

    def sample(self, rng: np.random.Generator) -> Sample:
        result = self._sample_fn(rng)
        if isinstance(result, tuple) and len(result) == 2:
            reward, info = result
        else:
            reward, info = result, {}
        info.setdefault("type", "custom")
        info.setdefault("name", self._name)
        return float(reward), info

    def expected_reward(self) -> Reward:
        if self._expected_reward_value is None:
            raise NotImplementedError("expected_reward is not available for this CustomArm")
        return float(self._expected_reward_value)

    def __repr__(self) -> str:
        er = self._expected_reward_value
        return f"CustomArm(name={self._name!r}, expected_reward={er!r})"
