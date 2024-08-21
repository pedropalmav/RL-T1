from enum import Enum

class ExperimentOption(Enum):
    EPSILON_GREEDY = 1
    OPTIMISTIC_INITIAL_VALUES = 2
    GRADIENT_BANDIT = 3