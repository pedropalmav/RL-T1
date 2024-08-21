import os
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from BanditResults import BanditResults
#from algorithms.IncrementalSimpleBandit import IncrementalSimpleBandit
from algorithms.GradientBandit import GradientBandit
from Plotter import Plotter
from ResultsWritter import ResultsWritter
from experiments.Experiment import Experiment
from ParamsReader import ParamsReader


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
    # Plotter.plot_average_results(experiment.results, "a) average_rewards")
    Plotter.plot_optimal_action_percentage(experiment.results, "f) optimal_action_percentage")

    # Gradient Bandit
    # params = [(0.1, True),
    #           (0.1, False),
    #           (0.4, True),
    #           (0.4, False)]
    
    # experiments_results = []
    # for param in params:    
    #     results = BanditResults()
    #     alpha, use_baseline = param
    #     for run_id in range(NUM_OF_RUNS):
    #         algorithm = GradientBandit(seed=run_id, alpha=alpha, use_baseline=use_baseline) 
    #         algorithm.run(NUM_OF_STEPS, results)
    #         results.save_current_run()
        
    #     # TODO: Create ExperimentResults class to store results and params
    #     experiments_results.append({"results": results, "params": [alpha]})
    # Plotter.plot_optimal_action_percentage(experiments_results, "f)"+" optimal_action_percentage")
    # # ResultsWritter.write_optimal_action_percentage(results, "optimal_action_percentage")
    # print("Plots saved to imgs folder")
