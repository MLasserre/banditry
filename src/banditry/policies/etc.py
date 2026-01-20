import numpy as np

class ETCPolicy:
    def __init__(self, bandit, m: int):
        self.__bandit = bandit
        self.__m = m
        self.__k = self.__bandit.n_arms
        self.__best_action = None
        self.__cur_step = 0  # Current step

        self.__Q = np.zeros(self.__k)  # Estimates of state-action values
        self.__N = np.zeros(self.__k)  # Number of times each action has been selected

        self.__list_rewards = []
        self.__list_actions = []

    def __select_action(self, step: int):
        if step < self.__m * self.__k:
            return self.__exploration(step)
        else:
            return self.__exploitation()
        
    def __exploration(self, step: int):
        return step % self.__k

    def __exploitation(self):
        if self.__best_action is None:
            max_value = np.max(self.__Q)
            candidates = np.flatnonzero(self.__Q == max_value)
            self.__best_action = int(np.random.choice(candidates))
        return self.__best_action
    
    def __update_values(self, action, reward):
        self.__N[action] += 1
        self.__Q[action] += (reward - self.__Q[action]) / self.__N[action]

    def learn(self, n_step: float):
        for i in range(n_step):
            action = self.__select_action(i)
            reward, _ = self.__bandit.pull(action)

            self.__update_values(action, reward)
            self.__list_actions.append(action)
            self.__list_rewards.append(reward)
            self.__cur_step += 1
            
        return self.__list_actions[-n_step:], self.__list_rewards[-n_step:]
