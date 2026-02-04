import pytest

from banditry import Bandit, BernoulliArm
from banditry.estimators import BaseEstimator, EWMeanEstimator, SampleMeanEstimator
from banditry.policies import ETCPolicy, EpsilonGreedyPolicy, UCBPolicy


class ConstantStepEstimator(BaseEstimator):
    def __init__(self, size: int, step_size: float, initial_estimates=None):
        super().__init__(size=size, initial_estimates=initial_estimates)
        self.step_size = float(step_size)

    def update(self, index: int, sample: float) -> None:
        self._counts[index] += 1
        self._estimates[index] += self.step_size * (sample - self._estimates[index])


def test_epsilon_greedy_uses_sample_mean_by_default():
    bandit = Bandit([BernoulliArm(1.0)], seed=0)
    policy = EpsilonGreedyPolicy(bandit, epsilon=0.0)
    assert isinstance(policy.estimator, SampleMeanEstimator)


def test_epsilon_greedy_supports_ewm_with_optimistic_initialization():
    bandit = Bandit([BernoulliArm(1.0)], seed=0)
    policy = EpsilonGreedyPolicy(
        bandit,
        epsilon=0.0,
        estimator="ew_mean",
        estimator_kwargs={"alpha": 0.2},
        initial_estimates=[5.0],
    )

    assert isinstance(policy.estimator, EWMeanEstimator)
    assert policy.value_estimates[0] == pytest.approx(5.0)
    policy.learn(1)
    assert policy.action_counts[0] == 1
    assert policy.value_estimates[0] == pytest.approx(4.2)


def test_epsilon_greedy_accepts_scalar_initial_estimate():
    bandit = Bandit([BernoulliArm(1.0), BernoulliArm(0.0)], seed=0)
    policy = EpsilonGreedyPolicy(
        bandit,
        epsilon=0.0,
        initial_estimates=3.0,
    )
    assert policy.value_estimates[0] == pytest.approx(3.0)
    assert policy.value_estimates[1] == pytest.approx(3.0)


def test_epsilon_greedy_supports_custom_estimator_class():
    bandit = Bandit([BernoulliArm(1.0)], seed=0)
    policy = EpsilonGreedyPolicy(
        bandit,
        epsilon=0.0,
        estimator=ConstantStepEstimator,
        estimator_kwargs={"step_size": 0.25},
        initial_estimates=[0.0],
    )
    policy.learn(1)
    assert policy.value_estimates[0] == pytest.approx(0.25)
    assert policy.action_counts[0] == 1


def test_epsilon_greedy_rejects_estimator_instances():
    bandit = Bandit([BernoulliArm(1.0)], seed=0)
    estimator_instance = EWMeanEstimator(size=1, alpha=0.1)
    with pytest.raises(TypeError):
        EpsilonGreedyPolicy(bandit, estimator=estimator_instance)


def test_epsilon_greedy_rejects_reserved_estimator_kwargs():
    bandit = Bandit([BernoulliArm(1.0)], seed=0)
    with pytest.raises(ValueError, match="reserved key"):
        EpsilonGreedyPolicy(bandit, estimator_kwargs={"size": 1})

    with pytest.raises(ValueError, match="reserved key"):
        EpsilonGreedyPolicy(bandit, estimator_kwargs={"initial_estimates": [0.0]})


def test_etc_and_ucb_use_sample_mean_estimators():
    bandit = Bandit([BernoulliArm(0.2), BernoulliArm(0.8)], seed=0)
    assert isinstance(ETCPolicy(bandit, m=1).estimator, SampleMeanEstimator)
    assert isinstance(UCBPolicy(bandit, delta=0.1).estimator, SampleMeanEstimator)


def test_ucb_delta_must_be_between_zero_and_one():
    bandit = Bandit([BernoulliArm(0.2), BernoulliArm(0.8)], seed=0)
    with pytest.raises(ValueError):
        UCBPolicy(bandit, delta=0.0)
    with pytest.raises(ValueError):
        UCBPolicy(bandit, delta=1.0)
