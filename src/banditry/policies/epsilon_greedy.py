import numpy as np


class EpsilonGreedyPolicy:
    def __init__(self, bandit, epsilon: float = 0.1, debug: bool = False):
        self.__epsilon = float(epsilon)  # Probability to explore
        self.__bandit = bandit
        self.__k = bandit.n_arms  # Number of actions
        self.__cur_step = 0  # Current step
        self.__um = lambda n: 1 / n  # Updating method (average sampling by default)

        self.__Q_1 = np.zeros(self.__k)  # Initial estimates of state-action values
        self.__Q = self.__Q_1.copy()  # Estimates of state-action values
        self.__N = np.zeros(self.__k)  # Number of times each action has been selected

        self.__list_rewards = []
        self.__list_actions = []

        self.__debug = debug
        self.__opt_act_taken = [] if debug else None

    def __select_action(self):
        is_greedy = np.random.uniform() > self.__epsilon
        return self.__exploitation() if is_greedy else self.__exploration()

    def __exploitation(self):
        # Break ties randomly among max actions
        max_value = np.max(self.__Q)
        candidates = np.flatnonzero(self.__Q == max_value)
        return int(np.random.choice(candidates))

    def __exploration(self):
        return int(np.random.randint(0, self.__k))

    def __update_values(self, action, reward):
        self.__N[action] = self.__N[action] + 1
        self.__Q[action] += self.__um(self.__N[action]) * (reward - self.__Q[action])

    def reset(self):
        self.__cur_step = 0
        self.__Q = self.__Q_1.copy()
        self.__N = np.zeros(self.__k)
        self.__list_actions.clear()
        self.__list_rewards.clear()
        if self.__debug:
            self.__opt_act_taken.clear()

    @property
    def current_step(self):
        return self.__cur_step

    def info(self):
        if not self.__debug:
            raise ValueError(f"debug has been set to {self.__debug}.")
        return self.__opt_act_taken, self.__list_rewards

    def set_initial_values(self, Q):
        if len(Q) != self.__k:
            raise ValueError(f"Length mismatch: Q({len(Q)}) must be of size {self.__k}.")
        self.__Q_1 = np.array(Q, dtype=float)
        if self.__cur_step == 0:
            self.__Q = self.__Q_1.copy()

    def set_epsilon(self, epsilon):
        if epsilon < 0 or epsilon > 1:
            raise ValueError("Value must be between 0 and 1 (inclusive)")
        self.__epsilon = float(epsilon)

    def set_update_method(self, method):
        self.reset()
        self.__um = method

    def learn(self, n_step):
        for _ in range(n_step):
            action = self.__select_action()
            reward, _ = self.__bandit.pull(action)
            self.__update_values(action, reward)
            self.__list_actions.append(action)
            self.__list_rewards.append(reward)
            self.__cur_step += 1

            if self.__debug:
                # Optional: log whether action matches bandit's known best if available
                best = getattr(self.__bandit, "best_action", None)
                self.__opt_act_taken.append(best is not None and action == best)

        return self.__list_actions[-n_step:], self.__list_rewards[-n_step:]
