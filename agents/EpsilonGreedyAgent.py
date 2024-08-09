import random
from agents.BaseAgent import BaseAgent

class EpsilonGreedyAgent(BaseAgent):
    def __init__(self, num_of_actions: int, epsilon: float):
        self.num_of_actions = num_of_actions
        self.__epsilon = epsilon
        self.q_values = [0] * num_of_actions
        self.action_counts = [0] * num_of_actions

    def get_action(self) -> int:
        if random.random() < self.__epsilon:
            return random.randrange(self.num_of_actions)
        else:
            max_q_value = max(self.q_values)
            return self.q_values.index(max_q_value)

    def learn(self, action, reward) -> None:
        self.action_counts[action] += 1
        self.q_values[action] = self.q_values[action] + (reward - self.q_values[action]) / self.action_counts[action]