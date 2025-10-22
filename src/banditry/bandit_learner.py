import numpy as np

class EpsilonGreedyLearner:
    def __init__(self,
                 bandit,
                 epsilon=0.1,
                 debug=False):

        self.__epsilon = epsilon # Probability to explore
        self.__k = bandit.getNActions() # Number of actions
        self.__cur_step = 0 # Current step
        self.__um = lambda N: 1/N # Updating method (average sampling by default)

        self.__Q_1 = np.zeros(self.__k) # Initial estimates of state-action values
        self.__Q = self.__Q_1 # Estimates of state-action values
        self.__N = np.zeros(self.__k) # Number of time each action has been selected

        self.__bandit = bandit # The bandit problem

        self.__list_rewards = []
        self.__list_actions = []

        self.__debug = debug

        if self.__debug:
            self.__opt_act_taken = []

    def __selectAction(self):
        is_greedy = np.random.uniform() > self.__epsilon
        if is_greedy:
            return self.__exploitation()
        else:
            return self.__exploration()

    def __exploitation(self):
        return np.argmax(self.__Q) # Would be better with random
    def __exploration(self):
        return np.random.randint(0, self.__k)

    def __updateValues(self, action, reward):
        self.__N[action] = self.__N[action] + 1
        self.__Q[action] += self.__um(self.__N[action]) * (reward - self.__Q[action]) 

    def reset(self):
        if self.__cur_step != 0:
            self.__cur_step = 0
            self.__Q = self.__Q_1
            self.__N = np.zeros(self.__k)

    def getCurrentStep(self):
        return self.__cur_step

    def getInfo(self):
        if self.__debug:
            return (self.__opt_act_taken, self.__list_rewards)
        else:
            raise ValueError(f"debug has been set to {self.__debug}.")

    def setInitalValues(self, Q):
        if len(Q) != self.__k:
            raise ValueError(f"Length mismatch: Q({len(Q)})"
                             f"must be of size {self.__k}.")
        self.__Q_1 = Q
        if self.__cur_step == 0:
            self.__Q = self.__Q_1

    def setEpsilon(self, epsilon):
        if epsilon < 0 or epsilon > 1:
            raise ValueError("Value must be between 0 and 1 (inclusive)")
        self.__epsilon = epsilon

    def setUpdateMethod(self, method):
        self.reset()
        self.__um = method

    def learn(self, n_step):
        for t in range(n_step):
            action = self.__selectAction()
            reward = self.__bandit.computeReward(action)
            self.__updateValues(action, reward)
            self.__list_actions.append(action)
            self.__list_rewards.append(reward)
            self.__cur_step += 1

            if self.__debug:
                self.__opt_act_taken.append(action == self.__bandit.getBestAction())

        return self.__list_actions[-n_step:], self.__list_rewards[-n_step:]
