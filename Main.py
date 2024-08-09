from BanditEnv import BanditEnv
from BanditResults import BanditResults
from agents.RandomAgent import RandomAgent
from agents.EpsilonGreedyAgent import EpsilonGreedyAgent
from algorithms.IncrementalSimpleBandit import IncrementalSimpleBandit


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
        algorithm = IncrementalSimpleBandit(seed=run_id, epsilon=0.1)
        algorithm.run(NUM_OF_STEPS, results)
        results.save_current_run()

    show_results(results)
