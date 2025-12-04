# Quickstart

## Installation

```bash
pip install banditry
```

For development (includes pytest):

```bash
pip install -e ".[dev]"
```

## Minimal example

Create a bandit with Bernoulli and Gaussian arms and run an epsilon-greedy policy:

```python
from banditry import Bandit, BernoulliArm, GaussianArm, EpsilonGreedyPolicy

arms = [
    BernoulliArm(0.2, name="bernoulli"),
    GaussianArm(mean=0.5, std=1.0, name="gaussian"),
]
bandit = Bandit(arms, seed=123)

policy = EpsilonGreedyPolicy(bandit, epsilon=0.1)
actions, rewards = policy.learn(10)
print(actions)
print(rewards)
```

## Custom arm

You can plug your own sampler via `CustomArm`:

```python
import numpy as np
from banditry import Bandit, CustomArm

exp_arm = CustomArm(lambda rng: rng.exponential(scale=2.0), expected_reward_value=2.0, name="exp")
bandit = Bandit([exp_arm], seed=0)
reward, info = bandit.pull(0)
print(reward, info)
```
