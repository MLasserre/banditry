from abc import ABC, abstractmethod
from typing import Optional
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
