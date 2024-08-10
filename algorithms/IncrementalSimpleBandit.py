from BanditEnv import BanditEnv
from BanditResults import BanditResults
from agents.EpsilonGreedyAgent import EpsilonGreedyAgent, ConstantStepAgent

class IncrementalSimpleBandit:
    def __init__(self, seed: int, epsilon: float, step_size: float):
        self.bandit = BanditEnv(seed=seed)
        self.num_of_arms = self.bandit.action_space
        if step_size != 0:
            self.agent = ConstantStepAgent(num_of_actions=self.num_of_arms, epsilon=epsilon, alpha = step_size)
        else:
            self.agent = EpsilonGreedyAgent(num_of_actions=self.num_of_arms, epsilon=epsilon)
        self.best_action = self.bandit.best_action

    def run(self, num_of_steps: int, results: BanditResults) -> None:
        for _ in range(num_of_steps):
            action = self.agent.get_action()
            reward = self.bandit.step(action)
            self.agent.learn(action, reward)
            is_best_action = action == self.best_action
            results.add_result(reward, is_best_action)