from BanditRun import BanditRun


class BanditResults:
    def __init__(self):
        self.__results = []
        self.__current_run = BanditRun()

    def add_result(self, reward: float, is_action_optimal: bool):
        self.__current_run.add_result(reward, is_action_optimal)

    def save_current_run(self):
        self.__results.append(self.__current_run)
        self.__current_run = BanditRun()

    def get_average_rewards(self) -> list[float]:
        return self.__get_average_statistic("reward")

    def __get_average_statistic(self, target_statistic: str):
        num_of_runs = len(self.__results)
        num_of_steps = self.__results[0].get_num_of_steps()
        average_statistic = []
        for step in range(num_of_steps):
            step_rewards = [run.get_average_statistics(target_statistic, step) for run in self.__results]
            average_statistic.append(sum(step_rewards) / num_of_runs)
        return average_statistic

    def get_optimal_action_percentage(self) -> list[float]:
        return self.__get_average_statistic("best_action")
