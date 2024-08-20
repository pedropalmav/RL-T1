from algorithms.BaseAlgorithm import BaseAlgorithm
from algorithms.GradientBandit import GradientBandit
from algorithms.IncrementalSimpleBandit import IncrementalSimpleBandit

class AlgorithmCatalog:
    
    @staticmethod
    def get_algorithm(name: str, params: dict) -> BaseAlgorithm:
        match(name):
            case "IncrementalSimpleBandit":
                return IncrementalSimpleBandit(seed=params["seed"], epsilon=params["epsilon"], bias=params["bias"], step_size=params["step_size"])
            case "GradientBandit":
                return GradientBandit(seed=params["seed"], alpha=params["alpha"], use_baseline=params["use_baseline"])
            case _:
                raise ValueError(f"Algorithm {name} not found")
