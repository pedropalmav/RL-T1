from abc import ABC, abstractmethod


class BaseAgent(ABC):

    @abstractmethod
    def get_action(self) -> int:
        pass

    @abstractmethod
    def learn(self, action, reward) -> None:
        pass
