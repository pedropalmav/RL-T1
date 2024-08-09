import random

from agents.BaseAgent import BaseAgent


class RandomAgent(BaseAgent):
    """
    This agent doesn't learn. It simply returns a random action.
    """

    def __init__(self, num_of_actions: int):
        self.num_of_actions = num_of_actions

    def get_action(self) -> int:
        return random.randrange(self.num_of_actions)

    def learn(self, action, reward) -> None:
        pass
