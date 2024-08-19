from abc import ABC, abstractmethod
from BanditResults import BanditResults

class BaseAlgorithm(ABC):
    @abstractmethod
    def run(self, num_of_steps: int, results: BanditResults) -> None:
        pass