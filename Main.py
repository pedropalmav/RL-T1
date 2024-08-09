from BanditEnv import BanditEnv
from BanditResults import BanditResults
from agents.RandomAgent import RandomAgent


def show_results(bandit_results: type(BanditResults)) -> None:
    print("\nAverage results")
    print("Step\tReward\tOptimal action (%)")
    average_rewards = bandit_results.get_average_rewards()
    optimal_action_percentage = bandit_results.get_optimal_action_percentage()
    for step in range(NUM_OF_STEPS):
        print(f"{step+1}\t{average_rewards[step]:0.3f}\t{optimal_action_percentage[step]:0.3f}")


if __name__ == "__main__":

    NUM_OF_RUNS = 2000
    NUM_OF_STEPS = 1000

    results = BanditResults()
    for run_id in range(NUM_OF_RUNS):
        bandit = BanditEnv(seed=run_id)
        num_of_arms = bandit.action_space
        agent = RandomAgent(num_of_arms)  # here you might change the agent that you want to use
        best_action = bandit.best_action
        for _ in range(NUM_OF_STEPS):
            action = agent.get_action()
            reward = bandit.step(action)
            agent.learn(action, reward)
            is_best_action = action == best_action
            results.add_result(reward, is_best_action)
        results.save_current_run()

    show_results(results)
