from BanditEnv import BanditEnv
from BanditResults import BanditResults
from agents.EpsilonGreedyAgent import EpsilonGreedyAgent, ConstantStepAgent
from algorithms.BaseAlgorithm import BaseAlgorithm

class IncrementalSimpleBandit(BaseAlgorithm):
    def __init__(self, seed: int, epsilon: float,  bias: float = 0, step_size: float = 0):
        self.bandit = BanditEnv(seed=seed)
        self.num_of_arms = self.bandit.action_space
        self.best_action = self.bandit.best_action
        if step_size != 0:
            self.agent = ConstantStepAgent(num_of_actions=self.num_of_arms, epsilon=epsilon, bias=bias, alpha=step_size)
        else:
            self.agent = EpsilonGreedyAgent(num_of_actions=self.num_of_arms, epsilon=epsilon, bias=bias)

    def run(self, num_of_steps: int, results: BanditResults) -> None:
        for _ in range(num_of_steps):
            action = self.agent.get_action()
            reward = self.bandit.step(action)
            self.agent.learn(action, reward)
            is_best_action = action == self.best_action
            results.add_result(reward, is_best_action)