from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple, Union, Sequence
import numpy as np
from .._types import Reward, Sample


class BaseArm(ABC):
    def __init__(self, name: Optional[str] = None):
        # Name can be supplied by the user; otherwise stays None and the bandit will assign a label.
        self._name = name
        self._num_pulls = 0

    @property
    def name(self) -> Optional[str]:
        return self._name

    @property
    def num_pulls(self) -> int:
        return self._num_pulls

    def _record_pull(self) -> None:
        self._num_pulls += 1

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


class ExponentialArm(BaseArm):
    def __init__(self, scale: float, name: Optional[str] = None):
        super().__init__(name)
        if scale <= 0:
            raise ValueError("scale must be positive.")
        self._scale = float(scale)

    def sample(self, rng: np.random.Generator) -> Sample:
        reward = rng.exponential(self._scale)
        return reward, {"type": "exponential", "scale": self._scale, "name": self._name}

    def expected_reward(self) -> Reward:
        return self._scale

    def __repr__(self) -> str:
        return f"ExponentialArm(scale={self._scale}, name={self._name!r})"


class PoissonArm(BaseArm):
    def __init__(self, rate: float, name: Optional[str] = None):
        super().__init__(name)
        if rate <= 0:
            raise ValueError("rate must be positive.")
        self._rate = float(rate)

    def sample(self, rng: np.random.Generator) -> Sample:
        reward = float(rng.poisson(self._rate))
        return reward, {"type": "poisson", "rate": self._rate, "name": self._name}

    def expected_reward(self) -> Reward:
        return self._rate

    def __repr__(self) -> str:
        return f"PoissonArm(rate={self._rate}, name={self._name!r})"


class UniformArm(BaseArm):
    def __init__(self, low: float, high: float, name: Optional[str] = None):
        super().__init__(name)
        if high <= low:
            raise ValueError("high must be greater than low.")
        self._low = float(low)
        self._high = float(high)

    def sample(self, rng: np.random.Generator) -> Sample:
        reward = float(rng.uniform(self._low, self._high))
        return reward, {"type": "uniform", "low": self._low, "high": self._high, "name": self._name}

    def expected_reward(self) -> Reward:
        return (self._low + self._high) / 2.0

    def __repr__(self) -> str:
        return f"UniformArm(low={self._low}, high={self._high}, name={self._name!r})"


class BetaArm(BaseArm):
    def __init__(self, alpha: float, beta: float, name: Optional[str] = None):
        super().__init__(name)
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta must be positive.")
        self._alpha = float(alpha)
        self._beta = float(beta)

    def sample(self, rng: np.random.Generator) -> Sample:
        reward = float(rng.beta(self._alpha, self._beta))
        return reward, {"type": "beta", "alpha": self._alpha, "beta": self._beta, "name": self._name}

    def expected_reward(self) -> Reward:
        return self._alpha / (self._alpha + self._beta)

    def __repr__(self) -> str:
        return f"BetaArm(alpha={self._alpha}, beta={self._beta}, name={self._name!r})"


class DriftingGaussianArm(BaseArm):
    def __init__(self, mean: float, std: float, drift_std: float, name: Optional[str] = None):
        super().__init__(name)
        if std <= 0:
            raise ValueError("std must be positive.")
        if drift_std < 0:
            raise ValueError("drift_std must be non-negative.")
        self._mu = float(mean)
        self._sigma = float(std)
        self._drift_std = float(drift_std)

    def sample(self, rng: np.random.Generator) -> Sample:
        reward = rng.normal(self._mu, self._sigma)
        info = {"type": "Gaussian", "mean": self._mu, "std": self._sigma, "name": self._name, "stationary": False}
        # Apply random walk drift on the mean
        if self._drift_std > 0:
            self._mu += rng.normal(0.0, self._drift_std)
        return reward, info

    def expected_reward(self) -> Reward:
        return self._mu

    def __repr__(self) -> str:
        return f"DriftingGaussianArm(mean={self._mu}, std={self._sigma}, drift_std={self._drift_std}, name={self._name!r})"


class PiecewiseBernoulliArm(BaseArm):
    """
    Bernoulli arm with scheduled probability changes.

    Schedule is a list of (pull_count, p) pairs. Progression uses BaseArm.num_pulls,
    which is incremented by Bandit.pull. Calling sample directly will not advance
    the schedule unless num_pulls is managed externally.
    """
    def __init__(self, schedule: Sequence[Tuple[int, float]], name: Optional[str] = None):
        super().__init__(name)
        if not schedule:
            raise ValueError("schedule must not be empty.")
        # Sort by step and validate probabilities
        sorted_sched = sorted(schedule, key=lambda x: x[0])
        for step, p in sorted_sched:
            if step < 0:
                raise ValueError("schedule steps must be non-negative.")
            if not (0.0 <= p <= 1.0):
                raise ValueError("probabilities must be in [0,1].")
        self._schedule = sorted_sched
        self._schedule_idx = 0
        self._p = float(sorted_sched[0][1])

    def _maybe_advance_schedule(self):
        # Use current num_pulls to determine if we should move to next p
        while (
            self._schedule_idx + 1 < len(self._schedule)
            and self.num_pulls >= self._schedule[self._schedule_idx + 1][0]
        ):
            self._schedule_idx += 1
            self._p = float(self._schedule[self._schedule_idx][1])

    def sample(self, rng: np.random.Generator) -> Sample:
        self._maybe_advance_schedule()
        r = float(rng.binomial(1, self._p))
        info = {"type": "bernoulli", "p": self._p, "name": self._name, "stationary": False}
        return r, info

    def expected_reward(self) -> Reward:
        return self._p

    def __repr__(self) -> str:
        return f"PiecewiseBernoulliArm(schedule={self._schedule}, name={self._name!r})"


class CustomArm(BaseArm):
    """
    Arm backed by a user-provided sampling function.

    sample_fn must accept an rng and return either:
      - a reward (float)
      - or a tuple (reward, info_dict)
    """

    def __init__(
        self,
        sample_fn: Callable[[np.random.Generator, Dict[str, Any]], Union[Reward, Tuple[Reward, dict]]],
        expected_reward_value: Optional[Union[float, Callable[[Dict[str, Any]], float]]] = None,
        update_fn: Optional[Callable[[np.random.Generator, Dict[str, Any], int], Dict[str, Any]]] = None,
        initial_state: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self._sample_fn = sample_fn
        self._expected_reward_value = expected_reward_value
        self._update_fn = update_fn
        self._state: Dict[str, Any] = initial_state.copy() if initial_state is not None else {}
        self._step = 0

    def _call_sample(self, rng: np.random.Generator):
        # Allow sample_fn to accept (rng, state) or just (rng)
        try:
            return self._sample_fn(rng, self._state)
        except TypeError:
            return self._sample_fn(rng)

    def sample(self, rng: np.random.Generator) -> Sample:
        result = self._call_sample(rng)
        if isinstance(result, tuple) and len(result) == 2:
            reward, info = result
        else:
            reward, info = result, {}
        info.setdefault("type", "custom")
        info.setdefault("name", self._name)
        info.setdefault("stationary", self.is_stationary)
        # Apply non-stationary update if any
        if self._update_fn is not None:
            self._state = self._update_fn(rng, self._state, self._step)
        self._step += 1
        return float(reward), info

    def expected_reward(self) -> Reward:
        if self._expected_reward_value is None:
            raise NotImplementedError("expected_reward is not available for this CustomArm")
        if callable(self._expected_reward_value):
            return float(self._expected_reward_value(self._state))
        return float(self._expected_reward_value)

    @property
    def is_stationary(self) -> bool:
        return self._update_fn is None

    def __repr__(self) -> str:
        er = self._expected_reward_value
        return f"CustomArm(name={self._name!r}, expected_reward={er!r})"
