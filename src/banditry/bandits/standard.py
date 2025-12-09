from typing import Optional, Sequence, Union, List

from .arms import BernoulliArm, GaussianArm, ExponentialArm, PoissonArm, UniformArm, BetaArm, BaseArm
from .bandit import Bandit


class BernoulliBandit(Bandit):
    """Convenience bandit composed of Bernoulli arms."""

    def __init__(self, probs: Sequence[float], labels: Optional[Sequence[str]] = None, seed: Optional[int] = None):
        if not probs:
            raise ValueError("At least one probability is required.")
        if labels is not None and len(labels) != len(probs):
            raise ValueError(f"labels must have length {len(probs)}, got {len(labels)}")
        arms: List[BaseArm] = []
        for idx, p in enumerate(probs):
            name = labels[idx] if labels is not None else None
            arms.append(BernoulliArm(p, name=name))
        super().__init__(arms, seed=seed)


class GaussianBandit(Bandit):
    """Convenience bandit composed of Gaussian arms."""

    def __init__(
        self,
        means: Sequence[float],
        stds: Optional[Union[float, Sequence[float]]] = 1.0,
        labels: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
    ):
        if not means:
            raise ValueError("At least one mean is required.")
        if labels is not None and len(labels) != len(means):
            raise ValueError(f"labels must have length {len(means)}, got {len(labels)}")

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
            name = labels[idx] if labels is not None else None
            arms.append(GaussianArm(mu, sigma, name=name))
        super().__init__(arms, seed=seed)


class ExponentialBandit(Bandit):
    """Convenience bandit composed of Exponential arms."""

    def __init__(self, scales: Sequence[float], labels: Optional[Sequence[str]] = None, seed: Optional[int] = None):
        if not scales:
            raise ValueError("At least one scale is required.")
        if labels is not None and len(labels) != len(scales):
            raise ValueError(f"labels must have length {len(scales)}, got {len(labels)}")
        arms: List[BaseArm] = []
        for idx, s in enumerate(scales):
            name = labels[idx] if labels is not None else None
            arms.append(ExponentialArm(s, name=name))
        super().__init__(arms, seed=seed)


class PoissonBandit(Bandit):
    """Convenience bandit composed of Poisson arms."""

    def __init__(self, rates: Sequence[float], labels: Optional[Sequence[str]] = None, seed: Optional[int] = None):
        if not rates:
            raise ValueError("At least one rate is required.")
        if labels is not None and len(labels) != len(rates):
            raise ValueError(f"labels must have length {len(rates)}, got {len(labels)}")
        arms: List[BaseArm] = []
        for idx, lam in enumerate(rates):
            name = labels[idx] if labels is not None else None
            arms.append(PoissonArm(lam, name=name))
        super().__init__(arms, seed=seed)


class UniformBandit(Bandit):
    """Convenience bandit composed of Uniform arms."""

    def __init__(self, lows: Sequence[float], highs: Sequence[float], labels: Optional[Sequence[str]] = None, seed: Optional[int] = None):
        if not lows or not highs or len(lows) != len(highs):
            raise ValueError("lows and highs must be non-empty and of equal length.")
        if labels is not None and len(labels) != len(lows):
            raise ValueError(f"labels must have length {len(lows)}, got {len(labels)}")
        arms: List[BaseArm] = []
        for idx, (lo, hi) in enumerate(zip(lows, highs)):
            name = labels[idx] if labels is not None else None
            arms.append(UniformArm(lo, hi, name=name))
        super().__init__(arms, seed=seed)


class BetaBandit(Bandit):
    """Convenience bandit composed of Beta arms."""

    def __init__(self, alphas: Sequence[float], betas: Sequence[float], labels: Optional[Sequence[str]] = None, seed: Optional[int] = None):
        if not alphas or not betas or len(alphas) != len(betas):
            raise ValueError("alphas and betas must be non-empty and of equal length.")
        if labels is not None and len(labels) != len(alphas):
            raise ValueError(f"labels must have length {len(alphas)}, got {len(labels)}")
        arms: List[BaseArm] = []
        for idx, (a, b) in enumerate(zip(alphas, betas)):
            name = labels[idx] if labels is not None else None
            arms.append(BetaArm(a, b, name=name))
        super().__init__(arms, seed=seed)
