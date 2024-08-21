import os

from BanditResults import BanditResults
from algorithms.AlgorithmCatalog import AlgorithmCatalog
from algorithms.AlgorithmOption import AlgorithmOption
from experiments.ExperimentOption import ExperimentOption
from ParamsReader import ParamsReader

class Experiment:

    def __init__(self, runs: int, steps: int):
        self.__runs = runs
        self.__steps = steps
        self.results = []
        self.setup_experiment()

    def setup_experiment(self) -> None:
        selected_option = self.select_experiment()
        self.params = ParamsReader.read_params(self.get_params_path(selected_option))
        self.algorithm = self.get_experiment_algorithm(selected_option)

    def select_experiment(self) -> ExperimentOption:
        print("Select experiment to run:")
        for option in ExperimentOption:
            print(f"{option.value}) {option.name.replace('_', ' ').title()}")
        selected_option = int(input("Option: "))
        return ExperimentOption(selected_option)

    def get_params_path(self, option: ExperimentOption) -> str:
        match(option):
            case ExperimentOption.EPSILON_GREEDY:
                return os.path.join("data", "epsilon_greedy_params.json")
            case ExperimentOption.OPTIMISTIC_INITIAL_VALUES:
                return os.path.join("data", "optimistic_value_params.json")
            case ExperimentOption.GRADIENT_BANDIT:
                return os.path.join("data", "gradient_bandit_params.json")
            case _:
                raise ValueError("Invalid option")
            
    def get_experiment_algorithm(self, option: ExperimentOption) -> AlgorithmOption:
        match(option):
            case ExperimentOption.EPSILON_GREEDY:
                return AlgorithmOption.INCREMENTAL_SIMPLE_BANDIT
            case ExperimentOption.OPTIMISTIC_INITIAL_VALUES:
                return AlgorithmOption.INCREMENTAL_SIMPLE_BANDIT
            case ExperimentOption.GRADIENT_BANDIT:
                return AlgorithmOption.GRADIENT_BANDIT
            case _:
                raise ValueError("Invalid option")

    def run(self) -> None:
        for triplet in self.params:
            # TODO: create a method for this loop
            results = BanditResults()
            for run_id in range(self.__runs):
                triplet["seed"] = run_id
                algorithm = AlgorithmCatalog.get_algorithm(self.algorithm, triplet)
                algorithm.run(self.__steps, results)
                results.save_current_run()
            # TODO: modify to add params showed in the plot
            self.results.append({"results": results, "params": triplet})