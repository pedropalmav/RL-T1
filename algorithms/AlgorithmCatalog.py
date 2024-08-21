from algorithms.BaseAlgorithm import BaseAlgorithm
from algorithms.GradientBandit import GradientBandit
from algorithms.IncrementalSimpleBandit import IncrementalSimpleBandit
from algorithms.AlgorithmOption import AlgorithmOption

class AlgorithmCatalog:
    
    @staticmethod
    def get_algorithm(name: str, params: dict) -> BaseAlgorithm:
        match(name):
            case AlgorithmOption.INCREMENTAL_SIMPLE_BANDIT:
                return IncrementalSimpleBandit(seed=params["seed"], epsilon=params["epsilon"], bias=params["bias"], step_size=params["step_size"])
            case AlgorithmOption.GRADIENT_BANDIT:
                return GradientBandit(seed=params["seed"], alpha=params["alpha"], use_baseline=params["use_baseline"])
            case _:
                raise ValueError(f"Algorithm {name} not found")
