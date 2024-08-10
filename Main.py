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
        plt.plot(experiment["results"].get_average_rewards(), 
                 label=f"$\\epsilon={experiment['params'][0]}$")
    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.legend()
    plt.savefig(os.path.join("imgs", f"{filename}.png"))

def plot_optimal_action_percentage(experiment_results: list[dict], filename: str = "optimal_action_percentage") -> None:
    plt.figure()
    for experiment in experiment_results:
        plt.plot(experiment["results"].get_optimal_action_percentage(),
                 label=f"$\\epsilon={experiment['params'][0]}$") #Q1={experiment['params'][1]}, 
    plt.xlabel("Steps")
    plt.ylabel("Optimal action (%)")
    plt.ylim(0, 1)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.legend()
    plt.savefig(os.path.join("imgs", f"{filename}.png"))


if __name__ == "__main__":

    NUM_OF_RUNS = 2000
    NUM_OF_STEPS = 1000
    PREFIX = "a)" # Assignment excercise prefix

    #triplets -> (epsilon, bias, step_size)
    params = [(0.1, 0, 0),
              (0.01, 0, 0),
              (0.0, 0, 0)]
    #params = [(0.1, 0, 0.1 ),
    #          (0.0, 5, 0.1)]
    
    experiments_results = []

    for triplet in params:
        results = BanditResults()
        epsilon, bias, step_size = triplet
        for run_id in range(NUM_OF_RUNS):
            algorithm = IncrementalSimpleBandit(seed=run_id, epsilon=epsilon, bias=bias, step_size=step_size) 
            algorithm.run(NUM_OF_STEPS, results)
            results.save_current_run()
        experiments_results.append({"results": results, "params": triplet})

    plot_average_results(experiments_results, PREFIX+" average_rewards")
    plot_optimal_action_percentage(experiments_results, PREFIX+" optimal_action_percentage")
    print("Plots saved in imgs folder")