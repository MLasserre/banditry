import numpy as np
import pytest

from banditry.bandits import BernoulliArm, GaussianArm


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
