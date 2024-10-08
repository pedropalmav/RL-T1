from BanditResults import BanditResults
from experiments.Experiment import Experiment


def show_results(bandit_results: BanditResults) -> None:
    print("\nAverage results")
    print("Step\tReward\tOptimal action (%)")
    average_rewards = bandit_results.get_average_rewards()
    optimal_action_percentage = bandit_results.get_optimal_action_percentage()
    for step in range(NUM_OF_STEPS):
        print(f"{step+1}\t{average_rewards[step]:0.3f}\t{optimal_action_percentage[step]:0.3f}")


if __name__ == "__main__":

    NUM_OF_RUNS = 2000
    NUM_OF_STEPS = 1000

    experiment = Experiment(runs=NUM_OF_RUNS, steps=NUM_OF_STEPS)
    experiment.run()
    experiment.plot_results()