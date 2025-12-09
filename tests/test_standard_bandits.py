import pytest

from banditry.bandits import (
    BernoulliBandit,
    GaussianBandit,
    ExponentialBandit,
    PoissonBandit,
    UniformBandit,
    BetaBandit,
)


def test_bernoulli_bandit_builds_with_labels():
    bandit = BernoulliBandit([0.1, 0.9], labels=["low", "high"], seed=0)
    assert bandit.n_arms == 2
    reward, info = bandit.pull("high")
    assert info["arm_label"] == "high"
    assert info["arm_index"] == 1
    assert reward in (0.0, 1.0)


def test_bernoulli_bandit_label_length_mismatch():
    with pytest.raises(ValueError):
        BernoulliBandit([0.1, 0.9], labels=["only_one"])


def test_gaussian_bandit_with_broadcast_std():
    bandit = GaussianBandit([0.0, 1.0], stds=0.5, labels=["g0", "g1"], seed=123)
    assert bandit.n_arms == 2
    _, info0 = bandit.pull("g0")
    _, info1 = bandit.pull("g1")
    assert info0["arm_label"] == "g0"
    assert info1["arm_label"] == "g1"
    assert info0["std"] == 0.5
    assert info1["std"] == 0.5


def test_gaussian_bandit_defaults_to_unit_std():
    bandit = GaussianBandit([0.0, 1.0], labels=["a", "b"], seed=1)
    _, info0 = bandit.pull("a")
    _, info1 = bandit.pull("b")
    assert info0["std"] == 1.0
    assert info1["std"] == 1.0


def test_gaussian_bandit_length_mismatch():
    with pytest.raises(ValueError):
        GaussianBandit([0.0, 1.0], stds=[0.5])
    with pytest.raises(ValueError):
        GaussianBandit([0.0], stds=[0.5, 0.6])


def test_exponential_bandit():
    bandit = ExponentialBandit([1.0, 2.0], labels=["e0", "e1"], seed=0)
    assert bandit.n_arms == 2
    _, info = bandit.pull("e0")
    assert info["arm_label"] == "e0"
    assert info["type"] == "exponential"


def test_poisson_bandit():
    bandit = PoissonBandit([1.0, 3.0], labels=["p0", "p1"], seed=0)
    assert bandit.n_arms == 2
    _, info = bandit.pull("p1")
    assert info["arm_label"] == "p1"
    assert info["type"] == "poisson"


def test_uniform_bandit():
    bandit = UniformBandit([0.0, -1.0], [1.0, 1.0], labels=["u0", "u1"], seed=0)
    _, info = bandit.pull("u0")
    assert info["arm_label"] == "u0"
    assert info["type"] == "uniform"


def test_beta_bandit():
    bandit = BetaBandit([2.0], [5.0], labels=["b0"], seed=0)
    _, info = bandit.pull("b0")
    assert info["arm_label"] == "b0"
    assert info["type"] == "beta"
