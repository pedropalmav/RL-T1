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
        self.select_experiment()
        self.experiment_params = ParamsReader.read_params(self.get_params_path(self.selected_experiment))
        self.algorithm = self.get_experiment_algorithm(self.selected_experiment)

    def select_experiment(self) -> None:
        print("Select experiment to run:")
        for option in ExperimentOption:
            print(f"{option.value}) {option.name.replace('_', ' ').title()}")
        selected_option = int(input("Option: "))
        self.selected_experiment = ExperimentOption(selected_option)

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
        for method_params in self.experiment_params:
            # TODO: create a method for this loop
            results = BanditResults()
            for run_id in range(self.__runs):
                method_params["seed"] = run_id
                algorithm = AlgorithmCatalog.get_algorithm(self.algorithm, method_params)
                algorithm.run(self.__steps, results)
                results.save_current_run()
            self.add_method_result(results, method_params)

    def add_method_result(self, results: BanditResults, params: dict) -> None:
        filter_keys = self.get_filter_keys()
        filtered_params = self.filter_params(params, filter_keys)
        self.results.append({"results": results, "params": filtered_params})

    def get_filter_keys(self) -> list:
        match(self.selected_experiment):
            case ExperimentOption.EPSILON_GREEDY:
                return ["epsilon"]
            case ExperimentOption.OPTIMISTIC_INITIAL_VALUES:
                return ["epsilon", "bias"]
            case ExperimentOption.GRADIENT_BANDIT:
                return ["alpha", "use_baseline"]
            case _:
                raise ValueError("Invalid option")

    def filter_params(self, params: dict, keys: list) -> dict:
        return {key: params[key] for key in keys}