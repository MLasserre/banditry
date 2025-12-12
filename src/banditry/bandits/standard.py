from typing import Optional, Sequence, Union, List, Tuple

from .arms import (
    BernoulliArm,
    GaussianArm,
    ExponentialArm,
    PoissonArm,
    UniformArm,
    BetaArm,
    DriftingGaussianArm,
    PiecewiseBernoulliArm,
    BaseArm,
)
from .bandit import Bandit


class BernoulliBandit(Bandit):
    """Convenience bandit composed of Bernoulli arms."""

    def __init__(self, probs: Sequence[float], names: Optional[Sequence[str]] = None, seed: Optional[int] = None):
        if not probs:
            raise ValueError("At least one probability is required.")
        if names is not None and len(names) != len(probs):
            raise ValueError(f"names must have length {len(probs)}, got {len(names)}")
        arms: List[BaseArm] = []
        for idx, p in enumerate(probs):
            name = names[idx] if names is not None else None
            arms.append(BernoulliArm(p, name=name))
        super().__init__(arms, seed=seed)


class GaussianBandit(Bandit):
    """Convenience bandit composed of Gaussian arms."""

    def __init__(
        self,
        means: Sequence[float],
        stds: Optional[Union[float, Sequence[float]]] = 1.0,
        names: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
    ):
        if not means:
            raise ValueError("At least one mean is required.")
        if names is not None and len(names) != len(means):
            raise ValueError(f"names must have length {len(means)}, got {len(names)}")

        # Broadcast stds if a single value is provided, default to 1.0 if None
        if stds is None or isinstance(stds, (int, float)):
            std_value = 1.0 if stds is None else float(stds)
            std_list = [std_value] * len(means)
        else:
            if len(stds) != len(means):
                raise ValueError(f"stds must have length {len(means)}, got {len(stds)}")
            std_list = list(stds)

        arms: List[BaseArm] = []
        for idx, (mu, sigma) in enumerate(zip(means, std_list)):
            name = names[idx] if names is not None else None
            arms.append(GaussianArm(mu, sigma, name=name))
        super().__init__(arms, seed=seed)


class DriftingGaussianBandit(Bandit):
    """Convenience bandit composed of drifting Gaussian arms."""

    def __init__(
        self,
        means: Sequence[float],
        stds: Optional[Union[float, Sequence[float]]] = 1.0,
        drift_stds: Optional[Union[float, Sequence[float]]] = 0.1,
        names: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
        restless: bool = False,
    ):
        if not means:
            raise ValueError("At least one mean is required.")
        if names is not None and len(names) != len(means):
            raise ValueError(f"names must have length {len(means)}, got {len(names)}")

        # Broadcast stds
        if stds is None or isinstance(stds, (int, float)):
            std_value = 1.0 if stds is None else float(stds)
            std_list = [std_value] * len(means)
        else:
            if len(stds) != len(means):
                raise ValueError(f"stds must have length {len(means)}, got {len(stds)}")
            std_list = list(stds)

        # Broadcast drift_stds
        if drift_stds is None or isinstance(drift_stds, (int, float)):
            drift_value = 0.0 if drift_stds is None else float(drift_stds)
            drift_list = [drift_value] * len(means)
        else:
            if len(drift_stds) != len(means):
                raise ValueError(f"drift_stds must have length {len(means)}, got {len(drift_stds)}")
            drift_list = list(drift_stds)

        arms: List[BaseArm] = []
        for idx, (mu, sigma, drift) in enumerate(zip(means, std_list, drift_list)):
            name = names[idx] if names is not None else None
            arms.append(DriftingGaussianArm(mu, sigma, drift, name=name))
        super().__init__(arms, seed=seed, restless=restless)


class ExponentialBandit(Bandit):
    """Convenience bandit composed of Exponential arms."""

    def __init__(self, scales: Sequence[float], names: Optional[Sequence[str]] = None, seed: Optional[int] = None):
        if not scales:
            raise ValueError("At least one scale is required.")
        if names is not None and len(names) != len(scales):
            raise ValueError(f"names must have length {len(scales)}, got {len(names)}")
        arms: List[BaseArm] = []
        for idx, s in enumerate(scales):
            name = names[idx] if names is not None else None
            arms.append(ExponentialArm(s, name=name))
        super().__init__(arms, seed=seed)


class PoissonBandit(Bandit):
    """Convenience bandit composed of Poisson arms."""

    def __init__(self, rates: Sequence[float], names: Optional[Sequence[str]] = None, seed: Optional[int] = None):
        if not rates:
            raise ValueError("At least one rate is required.")
        if names is not None and len(names) != len(rates):
            raise ValueError(f"names must have length {len(rates)}, got {len(names)}")
        arms: List[BaseArm] = []
        for idx, lam in enumerate(rates):
            name = names[idx] if names is not None else None
            arms.append(PoissonArm(lam, name=name))
        super().__init__(arms, seed=seed)


class UniformBandit(Bandit):
    """Convenience bandit composed of Uniform arms."""

    def __init__(self, lows: Sequence[float], highs: Sequence[float], names: Optional[Sequence[str]] = None, seed: Optional[int] = None):
        if not lows or not highs or len(lows) != len(highs):
            raise ValueError("lows and highs must be non-empty and of equal length.")
        if names is not None and len(names) != len(lows):
            raise ValueError(f"names must have length {len(lows)}, got {len(names)}")
        arms: List[BaseArm] = []
        for idx, (lo, hi) in enumerate(zip(lows, highs)):
            name = names[idx] if names is not None else None
            arms.append(UniformArm(lo, hi, name=name))
        super().__init__(arms, seed=seed)


class BetaBandit(Bandit):
    """Convenience bandit composed of Beta arms."""

    def __init__(self, alphas: Sequence[float], betas: Sequence[float], names: Optional[Sequence[str]] = None, seed: Optional[int] = None):
        if not alphas or not betas or len(alphas) != len(betas):
            raise ValueError("alphas and betas must be non-empty and of equal length.")
        if names is not None and len(names) != len(alphas):
            raise ValueError(f"names must have length {len(alphas)}, got {len(names)}")
        arms: List[BaseArm] = []
        for idx, (a, b) in enumerate(zip(alphas, betas)):
            name = names[idx] if names is not None else None
            arms.append(BetaArm(a, b, name=name))
        super().__init__(arms, seed=seed)


class PiecewiseBernoulliBandit(Bandit):
    """Convenience bandit composed of piecewise Bernoulli arms."""

    def __init__(
        self,
        schedules: Sequence[Sequence[Tuple[int, float]]],
        names: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
        restless: bool = False,
    ):
        if not schedules:
            raise ValueError("At least one schedule is required.")
        if names is not None and len(names) != len(schedules):
            raise ValueError(f"names must have length {len(schedules)}, got {len(names)}")
        arms: List[BaseArm] = []
        for idx, schedule in enumerate(schedules):
            name = names[idx] if names is not None else None
            arms.append(PiecewiseBernoulliArm(schedule=schedule, name=name))
        super().__init__(arms, seed=seed, restless=restless)
