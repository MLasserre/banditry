import pytest

from banditry.bandits import BernoulliBandit, GaussianBandit


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
