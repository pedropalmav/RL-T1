import numpy as np
from agents.BaseAgent import BaseAgent

class SGAgent(BaseAgent):
    def __init__(self, num_of_actions, alpha) -> int:
        self.num_of_actions = num_of_actions
        self.baseline = 0
        self.H = np.zeros(num_of_actions)
        self.__alpha = alpha
        self.__policy = self.update_policy()

    def update_policy(self) -> np.ndarray:
        C = np.max(self.H)
        policy =  np.exp(self.H - C) / np.sum(np.exp(self.H - C))
        return policy

    def get_action(self) -> int:
        return np.argmax(self.__policy)

    def learn(self, action, reward) -> None:
        for a in range(self.num_of_actions):
            if a == action:
                self.H[a] += self.__alpha * (reward - self.baseline) * (1 - self.__policy[a])
            else:
                self.H[a] += self.__alpha * (reward - self.baseline) * -1 * self.__policy[a]
            self.__policy = self.update_policy()

    def update_baseline(self, reward, step) -> None:
        self.baseline += (reward - self.baseline) / step
