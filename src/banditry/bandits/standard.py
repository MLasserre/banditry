from typing import Optional, Sequence, Union, List

from .arms import BernoulliArm, GaussianArm, BaseArm
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
