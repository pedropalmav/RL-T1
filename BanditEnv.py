import numpy as np


class BanditEnv:

    def __init__(self, seed: int, mean: float = 0.0, num_of_arms: int = 10):
        assert num_of_arms > 0, "num_of_arms must be greater than 0"
        self.__num_of_arms = num_of_arms
        rng = np.random.RandomState(seed)
        self.__means = rng.normal(loc=mean, scale=1.0, size=self.__num_of_arms)

    def step(self, action: int) -> float:
        return np.random.normal(self.__means[action])

    @property
    def best_action(self) -> int:
        return int(self.__means.argmax())

    @property
    def action_space(self) -> int:
        return self.__num_of_arms
