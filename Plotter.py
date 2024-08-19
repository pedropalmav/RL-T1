import os
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

class Plotter:
    @staticmethod
    def plot_average_results(experiment_results: list[dict], filename: str = "average_rewards") -> None:
        plt.figure()
        for experiment in experiment_results:
            plt.plot(experiment["results"].get_average_rewards(), 
                    label=f"$\\epsilon={experiment['params'][0]}$")
        plt.xlabel("Steps")
        plt.ylabel("Average reward")
        plt.legend()
        plt.savefig(os.path.join("imgs", f"{filename}.png"))

    @staticmethod
    def plot_optimal_action_percentage(experiment_results: list[dict], filename: str = "optimal_action_percentage") -> None:
        plt.figure()
        for experiment in experiment_results:
            plt.plot(experiment["results"].get_optimal_action_percentage(),
                    label=f"$\\epsilon={experiment['params'][0]}$")
        plt.xlabel("Steps")
        plt.ylabel("Optimal action (%)")
        plt.ylim(0, 1)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.legend()
        plt.savefig(os.path.join("imgs", f"{filename}.png"))