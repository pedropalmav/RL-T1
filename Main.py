import os
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from BanditResults import BanditResults
from algorithms.IncrementalSimpleBandit import IncrementalSimpleBandit


def show_results(bandit_results: BanditResults) -> None:
    print("\nAverage results")
    print("Step\tReward\tOptimal action (%)")
    average_rewards = bandit_results.get_average_rewards()
    optimal_action_percentage = bandit_results.get_optimal_action_percentage()
    for step in range(NUM_OF_STEPS):
        print(f"{step+1}\t{average_rewards[step]:0.3f}\t{optimal_action_percentage[step]:0.3f}")

def plot_average_results(experiment_results: list[dict], filename: str = "average_rewards") -> None:
    plt.figure()
    for experiment in experiment_results:
        plt.plot(experiment["results"].get_average_rewards(), label=f"$\\epsilon={experiment['epsilon']}$")
    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.legend()
    plt.savefig(os.path.join("imgs", f"{filename}.png"))

def plot_optimal_action_percentage(experiment_results: list[dict], filename: str = "optimal_action_percentage") -> None:
    plt.figure()
    for experiment in experiment_results:
        plt.plot(experiment["results"].get_optimal_action_percentage(), label=f"$\\epsilon={experiment['epsilon']}$")
    plt.xlabel("Steps")
    plt.ylabel("Optimal action (%)")
    plt.ylim(0, 1)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.legend()
    plt.savefig(os.path.join("imgs", f"{filename}.png"))


if __name__ == "__main__":

    NUM_OF_RUNS = 2000
    NUM_OF_STEPS = 1000

    epsilons = [0.1, 0.01, 0.0]
    experiments_results = []

    for epsilon in epsilons:
        results = BanditResults()
        for run_id in range(NUM_OF_RUNS):
            algorithm = IncrementalSimpleBandit(seed=run_id, epsilon=epsilon)
            algorithm.run(NUM_OF_STEPS, results)
            results.save_current_run()
        experiments_results.append({"results": results, "epsilon": epsilon})

    plot_average_results(experiments_results)
    plot_optimal_action_percentage(experiments_results)