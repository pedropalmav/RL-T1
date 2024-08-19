import numpy as np
from agents.BaseAgent import BaseAgent

class SGAgent(BaseAgent):
    def __init__(self, num_of_actions, alpha):
        self.num_of_actions = num_of_actions
        self.alpha = alpha
        self.H = np.zeros(num_of_actions)
        self.__policy = self.update_policy()
        self.baseline = 0
    

    def get_action(self) -> int:
        return np.random.choice(range(self.num_of_actions), p=self.__policy)
    
    def update_policy(self)-> np.ndarray:
        C = np.max(self.H)
        return np.exp(self.H - C) / np.sum(np.exp(self.H - C))
    
    def update_baseline(self, reward, step) -> None:
        self.baseline +=  (reward - self.baseline) / step
    
    def learn(self, action, reward) -> None:
        for a in range(self.num_of_actions):
            self.H[a] += reward + self.alpha * (reward - self.baseline) * (int(a == action) - self.__policy[a])
        self.__policy = self.update_policy()