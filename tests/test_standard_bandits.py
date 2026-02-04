import numpy as np
import pytest

from banditry.bandits import (
    BernoulliBandit,
    GaussianBandit,
    DriftingGaussianBandit,
    ExponentialBandit,
    PoissonBandit,
    UniformBandit,
    BetaBandit,
    PiecewiseBernoulliBandit,
)


def test_bernoulli_bandit_builds_with_names():
    bandit = BernoulliBandit([0.1, 0.9], names=["low", "high"], seed=0)
    assert bandit.n_arms == 2
    reward, info = bandit.pull("high")
    assert info["arm_name"] == "high"
    assert info["arm_index"] == 1
    assert reward in (0.0, 1.0)


def test_bernoulli_bandit_name_length_mismatch():
    with pytest.raises(ValueError):
        BernoulliBandit([0.1, 0.9], names=["only_one"])


def test_gaussian_bandit_with_broadcast_std():
    bandit = GaussianBandit([0.0, 1.0], stds=0.5, names=["g0", "g1"], seed=123)
    assert bandit.n_arms == 2
    _, info0 = bandit.pull("g0")
    _, info1 = bandit.pull("g1")
    assert info0["arm_name"] == "g0"
    assert info1["arm_name"] == "g1"
    assert info0["std"] == 0.5
    assert info1["std"] == 0.5


def test_gaussian_bandit_defaults_to_unit_std():
    bandit = GaussianBandit([0.0, 1.0], names=["a", "b"], seed=1)
    _, info0 = bandit.pull("a")
    _, info1 = bandit.pull("b")
    assert info0["std"] == 1.0
    assert info1["std"] == 1.0


def test_gaussian_bandit_length_mismatch():
    with pytest.raises(ValueError):
        GaussianBandit([0.0, 1.0], stds=[0.5])
    with pytest.raises(ValueError):
        GaussianBandit([0.0], stds=[0.5, 0.6])


def test_drifting_gaussian_bandit_builds_with_broadcast():
    bandit = DriftingGaussianBandit([0.0, 1.0], stds=0.5, drift_stds=[0.1, 0.2], names=["d0", "d1"], seed=0)
    assert bandit.n_arms == 2
    _, info0 = bandit.pull("d0")
    assert info0["arm_name"] == "d0"
    assert info0["type"] == "Gaussian"


def test_exponential_bandit():
    bandit = ExponentialBandit([1.0, 2.0], names=["e0", "e1"], seed=0)
    assert bandit.n_arms == 2
    _, info = bandit.pull("e0")
    assert info["arm_name"] == "e0"
    assert info["type"] == "exponential"


def test_poisson_bandit():
    bandit = PoissonBandit([1.0, 3.0], names=["p0", "p1"], seed=0)
    assert bandit.n_arms == 2
    _, info = bandit.pull("p1")
    assert info["arm_name"] == "p1"
    assert info["type"] == "poisson"


def test_uniform_bandit():
    bandit = UniformBandit([0.0, -1.0], [1.0, 1.0], names=["u0", "u1"], seed=0)
    _, info = bandit.pull("u0")
    assert info["arm_name"] == "u0"
    assert info["type"] == "uniform"


def test_beta_bandit():
    bandit = BetaBandit([2.0], [5.0], names=["b0"], seed=0)
    _, info = bandit.pull("b0")
    assert info["arm_name"] == "b0"
    assert info["type"] == "beta"


def test_piecewise_bernoulli_bandit_schedule():
    bandit = PiecewiseBernoulliBandit([[(0, 0.1), (2, 0.9)]], names=["pw"], seed=0)
    _, info1 = bandit.pull("pw")
    assert info1["p"] == 0.1
    _, info2 = bandit.pull("pw")
    assert info2["p"] == 0.1
    _, info3 = bandit.pull("pw")
    assert info3["p"] == 0.9


def test_standard_bandit_rejects_rng_and_seed_together():
    with pytest.raises(ValueError, match="either rng or seed"):
        BernoulliBandit([0.5], rng=np.random.default_rng(0), seed=0)


def test_standard_bandit_accepts_rng_for_reproducibility():
    bandit_1 = BernoulliBandit([0.5], rng=np.random.default_rng(123))
    bandit_2 = BernoulliBandit([0.5], rng=np.random.default_rng(123))
    rewards_1 = [bandit_1.pull(0)[0] for _ in range(8)]
    rewards_2 = [bandit_2.pull(0)[0] for _ in range(8)]
    assert rewards_1 == rewards_2
