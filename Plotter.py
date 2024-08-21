import os
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

class Plotter:
    @staticmethod
    def plot_average_results(experiment_results: list[dict], filename: str = "average_rewards") -> None:
        plt.figure()
        for experiment in experiment_results:
            label = Plotter.generate_label(experiment["params"])
            plt.plot(experiment["results"].get_average_rewards(), 
                    label=label)
        plt.xlabel("Steps")
        plt.ylabel("Average reward")
        plt.legend()
        plt.savefig(os.path.join("imgs", f"{filename}.png"))

    @staticmethod
    def plot_optimal_action_percentage(experiment_results: list[dict], filename: str = "optimal_action_percentage") -> None:
        plt.figure()
        for experiment in experiment_results:
            label = Plotter.generate_label(experiment["params"])
            plt.plot(experiment["results"].get_optimal_action_percentage(),
                    label=label)
        plt.xlabel("Steps")
        plt.ylabel("Optimal action (%)")
        plt.ylim(0, 1)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.legend()
        plt.savefig(os.path.join("imgs", f"{filename}.png"))

    @staticmethod
    def generate_label(params: dict) -> str:
        label = ""
        for key, value in params.items():
            label += Plotter.label_for_param(key, value)
        return label[:-2]
    
    @staticmethod
    def label_for_param(key: str, value: float) -> str:
        if key == "bias":
            return f"$Q_1={value}$, "
        if key == "epsilon" or key == "alpha":
            return f"$\\{key}={value}$, "
        if key == "baseline":
            return f"{'with' if value else 'without'} baseline, "
        return ""