class BanditRun:

    def __init__(self):
        self.__reward_history = []
        self.__best_action_history = []

    def add_result(self, reward: float, is_best_action: bool) -> None:
        self.__reward_history.append(reward)
        self.__best_action_history.append(int(is_best_action))

    def get_average_statistics(self, statistic: str, step: int) -> float:
        assert statistic in ["reward", "best_action"], "statistic must be 'reward' or 'best_action'"
        if statistic == "reward":
            return self.__reward_history[step]
        return self.__best_action_history[step]

    def get_num_of_steps(self) -> int:
        return len(self.__reward_history)
