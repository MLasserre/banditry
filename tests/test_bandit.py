import pytest

from banditry.bandits import Bandit, BernoulliArm, GaussianArm, DriftingGaussianArm, PiecewiseBernoulliArm


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
    assert arms[0].num_pulls == 1
    assert arms[1].num_pulls == 1
    assert bandit.arm_num_pulls("A") == 1
    assert bandit.arm_num_pulls(0) == 1


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


def test_bandit_evolve_runs_on_pulled_arm():
    # Evolve function increments a counter on each pull
    class CounterArm(BernoulliArm):
        def __init__(self, p, name=None):
            super().__init__(p, name)
            self.count = 0

        @property
        def is_stationary(self):
            return False

        def evolve(self, rng, step: int):
            self.count += 1

    a0 = CounterArm(1.0, name="A")
    a1 = CounterArm(0.0, name="B")
    bandit = Bandit([a0, a1], seed=0, restless=False)
    bandit.pull("A")
    bandit.pull("B")
    assert a0.count == 1
    assert a1.count == 1


def test_repr_contains_labels_and_counts():
    arms = [GaussianArm(0.0, 1.0, name="g0"), GaussianArm(1.0, 1.0, name="g1")]
    bandit = Bandit(arms, seed=42)
    rep = repr(bandit)
    assert "Bandit" in rep
    assert "n_arms=2" in rep
    assert "num_pulls=0" in rep
    assert "g0" in rep and "g1" in rep


def test_restless_bandit_advances_idle_non_stationary_arm():
    drifting = DriftingGaussianArm(mean=0.0, std=1.0, drift_std=0.5, name="dg")
    stationary = BernoulliArm(1.0, name="stable")
    bandit = Bandit([drifting, stationary], seed=0, restless=True)
    mu_before = drifting.expected_reward()
    # Pull only the stationary arm; drifting arm should still evolve while idle
    bandit.pull("stable")
    mu_after = drifting.expected_reward()
    assert mu_after != mu_before


def test_rested_bandit_leaves_idle_arm_unchanged():
    drifting = DriftingGaussianArm(mean=0.0, std=1.0, drift_std=0.5, name="dg")
    stationary = BernoulliArm(1.0, name="stable")
    bandit = Bandit([drifting, stationary], seed=0, restless=False)
    mu_before = drifting.expected_reward()
    bandit.pull("stable")
    # Without restlessness, the drifting arm should not evolve when idle
    assert drifting.expected_reward() == mu_before


def test_restless_bandit_advances_piecewise_schedule_without_pull():
    pw = PiecewiseBernoulliArm(schedule=[(0, 0.1), (1, 0.9)], name="pw")
    stationary = BernoulliArm(1.0, name="stable")
    bandit = Bandit([pw, stationary], seed=0, restless=True)
    bandit.pull("stable")  # advances pw via restlessness
    _, info_pw = bandit.pull("pw")
    assert info_pw["p"] == 0.9
