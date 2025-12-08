import numpy as np
import pytest

from banditry.bandits import BernoulliArm, GaussianArm, CustomArm


def test_bernoulli_arm_sample_and_expected_reward():
    arm = BernoulliArm(0.7, name="coin")
    rng = np.random.default_rng(0)
    reward, info = arm.sample(rng)
    assert reward in (0.0, 1.0)
    assert info["type"] == "bernoulli"
    assert info["p"] == 0.7
    assert info["name"] == "coin"
    assert arm.expected_reward() == pytest.approx(0.7)


def test_bernoulli_invalid_p():
    with pytest.raises(ValueError):
        BernoulliArm(-0.1)
    with pytest.raises(ValueError):
        BernoulliArm(1.1)


def test_gaussian_arm_sample_and_expected_reward():
    arm = GaussianArm(mean=1.5, std=0.5, name="g")
    rng = np.random.default_rng(0)
    reward, info = arm.sample(rng)
    assert isinstance(reward, float)
    assert info["type"] == "Gaussian"
    assert info["mean"] == 1.5
    assert info["std"] == 0.5
    assert info["name"] == "g"
    assert arm.expected_reward() == pytest.approx(1.5)


def test_gaussian_invalid_std():
    with pytest.raises(ValueError):
        GaussianArm(mean=0.0, std=0.0)
    with pytest.raises(ValueError):
        GaussianArm(mean=0.0, std=-1.0)


def test_custom_arm_with_and_without_expected_reward():
    rng = np.random.default_rng(0)

    def sampler(r):
        return r.normal()

    arm_unknown = CustomArm(sample_fn=sampler, expected_reward_value=None, name="u")
    reward, info = arm_unknown.sample(rng)
    assert isinstance(reward, float)
    assert info["type"] == "custom"
    assert info["name"] == "u"
    with pytest.raises(NotImplementedError):
        arm_unknown.expected_reward()

    arm_known = CustomArm(sample_fn=sampler, expected_reward_value=0.5, name="k")
    reward2, info2 = arm_known.sample(rng)
    assert isinstance(reward2, float)
    assert arm_known.expected_reward() == pytest.approx(0.5)


def test_custom_arm_non_stationary_update():
    rng = np.random.default_rng(0)

    def sampler(r, state):
        mu = state.get("mu", 0.0)
        return r.normal(mu, 1.0)

    def updater(r, state, step):
        new_state = dict(state)
        new_state["mu"] = new_state.get("mu", 0.0) + 0.1
        return new_state

    arm = CustomArm(sample_fn=sampler, update_fn=updater, expected_reward_value=lambda st: st.get("mu", 0.0), initial_state={"mu": 0.0}, name="drift")
    _, info1 = arm.sample(rng)
    assert info1["stationary"] is False
    mu_after_first = arm.expected_reward()
    assert mu_after_first == pytest.approx(0.1)
    arm.sample(rng)
    assert arm.expected_reward() == pytest.approx(0.2)
