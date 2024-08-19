import os
from BanditResults import BanditResults

class ResultsWritter:

    # TODO: Replace BanditResults with ExperimentResults
    # TODO: Format writting for json files
    @staticmethod
    def write_average_rewards(results: BanditResults, filename: str):
        with open(os.path.join("results", f"{filename}.json"), "w") as file:
            data = results.get_average_rewards()
            for line in data:
                print(line, file=file)

    @staticmethod
    def write_optimal_action_percentage(results: BanditResults, filename: str):
        with open(os.path.join("results", f"{filename}.json"), "w") as file:
            data = results.get_optimal_action_percentage()
            for line in data:
                print(line, file=file)