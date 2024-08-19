from BanditEnv import BanditEnv
from BanditResults import BanditResults
from agents.SGAgent import SGAgent
from algorithms.BaseAlgorithm import BaseAlgorithm

class GradientBandit(BaseAlgorithm):
    def __init__(self, seed: int, alpha: float, use_baseline: bool = False):
        self.bandit = BanditEnv(seed=seed, mean = 4.0)
        self.num_of_arms = self.bandit.action_space
        self.agent = SGAgent(num_of_actions=self.num_of_arms, alpha=alpha)
        self.best_action = self.bandit.best_action
        self.use_baseline = use_baseline

    def run(self, num_of_steps: int, results: BanditResults) -> None:
        for step in range(num_of_steps):
            #self.agent.update_policy()
            action = self.agent.get_action()
            reward = self.bandit.step(action)
            self.agent.learn(action, reward)
            if self.use_baseline:
               self.agent.update_baseline(reward, step + 1)
            is_best_action = action == self.best_action
            results.add_result(reward, is_best_action)