import random
from agents.BaseAgent import BaseAgent

class EpsilonGreedyAgent(BaseAgent):
    def __init__(self, num_of_actions: int, epsilon: float, bias: int):
        self.num_of_actions = num_of_actions
        self.__epsilon = epsilon
        self.q_values = [bias] * num_of_actions
        self.action_counts = [0] * num_of_actions

    def get_action(self) -> int:
        if random.random() < self.__epsilon:
            return random.randrange(self.num_of_actions)
        else:
            max_q_value = max(self.q_values)
            max_idx = [i for i, q in enumerate(self.q_values) if q == max_q_value]
            return random.choice(max_idx)

    def learn(self, action, reward) -> None:
        self.action_counts[action] += 1
        self.q_values[action] = self.q_values[action] + (reward - self.q_values[action]) / self.action_counts[action]

class ConstantStepAgent(BaseAgent):
    def __init__(self, num_of_actions: int, epsilon: float, bias: int ,alpha: float):
        self.num_of_actions = num_of_actions
        self.__epsilon = epsilon
        self.q_values = [bias] * num_of_actions
        self.action_counts = [0] * num_of_actions
        self.__alpha = alpha

    def get_action(self) -> int:
        if random.random() < self.__epsilon:
            return random.randrange(self.num_of_actions)
        else:
            max_q_value = max(self.q_values)
            max_idx = [i for i, q in enumerate(self.q_values) if q == max_q_value]
            return random.choice(max_idx)

    def learn(self, action, reward) -> None:
        self.action_counts[action] += 1
        self.q_values[action] = self.q_values[action] + self.__alpha * (reward - self.q_values[action]) 

import numpy as np
class SGAgent(BaseAgent):
    def __init__(self, num_of_actions, alpha) -> int:
        self.num_of_actions = num_of_actions
        self.H = [0] * num_of_actions
        self.action_counts = [0] * num_of_actions
        self.__alpha = alpha

    def get_action(self) -> int:
        C = max(self.H)
        policy =  np.exp(self.H - C) / np.sum(np.exp(self.H - C))
        return np.argmax(policy)

    def learn(self, action, reward) -> None:
        for a in range(self.num_of_actions):
            if a == action:
                self.H[a] += self.__alpha * reward * (1 - policy[a])
            else:
                self.H[a] += self.__alpha * reward * (- policy[a])