import os
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from BanditResults import BanditResults
#from algorithms.IncrementalSimpleBandit import IncrementalSimpleBandit
from algorithms.GradientBandit import GradientBandit
from Plotter import Plotter


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
    """
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
    """
    results = BanditResults()
    alpha = 0.1
    use_baseline = True
    experiments_results = []
    for run_id in range(NUM_OF_RUNS):
        algorithm = GradientBandit(seed=run_id, alpha=alpha, use_baseline=use_baseline) 
        algorithm.run(NUM_OF_STEPS, results)
        results.save_current_run()
    experiments_results.append({"results": results, "params": [alpha]})
    Plotter.plot_optimal_action_percentage(experiments_results, "f)"+" optimal_action_percentage")
    print("Plots saved to imgs folder")
