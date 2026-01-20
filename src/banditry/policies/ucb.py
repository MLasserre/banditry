import numpy as np

class UCBPolicy:
    def __init__(self, bandit, delta: int):
        self.__bandit = bandit
        self.__delta = delta
        self.__k = self.__bandit.n_arms
        self.__cur_step = 0  # Current step

        self.__Q = np.zeros(self.__k)  # Estimates of state-action values
        self.__bonus = np.full(self.__k, np.inf)
        self.__N = np.zeros(self.__k)  # Number of times each action has been selected

        self.__list_rewards = []
        self.__list_actions = []

    def __select_action(self):
        UCB = self.__Q + self.__bonus
        max_value = np.max(UCB)
        candidates = np.flatnonzero(UCB == max_value)
        return int(np.random.choice(candidates))
    
    def __update_values(self, action, reward):
        self.__N[action] += 1
        self.__Q[action] += (reward - self.__Q[action]) / self.__N[action]
        self.__bonus[action] = np.sqrt(2*np.log(1/self.__delta)/self.__N[action])

    def learn(self, n_step: float):
        for i in range(n_step):
            action = self.__select_action()
            reward, _ = self.__bandit.pull(action)

            self.__update_values(action, reward)
            self.__list_actions.append(action)
            self.__list_rewards.append(reward)
            self.__cur_step += 1

        return self.__list_actions[-n_step:], self.__list_rewards[-n_step:]