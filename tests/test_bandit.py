import pytest

from banditry.bandits import Bandit, BernoulliArm, GaussianArm


def test_default_labels_are_assigned_and_unique():
    arms = [BernoulliArm(0.2, "A"), BernoulliArm(0.8)]
    bandit = Bandit(arms, seed=0)

    assert bandit.n_arms == 2
    # Default labels come from positional naming
    reward0, info0 = bandit.pull(0)
    assert info0["arm_index"] == 0
    assert info0["arm_label"] == "A"
    reward1, info1 = bandit.pull("arm1")
    assert info1["arm_index"] == 1
    assert info1["arm_label"] == "arm1"
    assert bandit.num_pulls == 2
    assert reward0 in (0.0, 1.0)
    assert reward1 in (0.0, 1.0)


def test_named_arms_and_duplicate_label_error():
    a = BernoulliArm(1.0, name="A")
    b = BernoulliArm(0.0, name="B")
    bandit = Bandit([a, b], seed=123)
    # Pull by name and ensure metadata matches
    reward, info = bandit.pull("A")
    assert reward == 1.0
    assert info["arm_index"] == 0
    assert info["arm_label"] == "A"

    # Duplicate labels should raise
    with pytest.raises(ValueError):
        Bandit([BernoulliArm(0.5, name="dup"), BernoulliArm(0.5, name="dup")])


def test_repr_contains_labels_and_counts():
    arms = [GaussianArm(0.0, 1.0, name="g0"), GaussianArm(1.0, 1.0, name="g1")]
    bandit = Bandit(arms, seed=42)
    rep = repr(bandit)
    assert "Bandit" in rep
    assert "n_arms=2" in rep
    assert "num_pulls=0" in rep
    assert "g0" in rep and "g1" in rep
