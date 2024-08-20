from BanditResults import BanditResults
from algorithms.AlgorithmCatalog import AlgorithmCatalog
class Experiment:

    def __init__(self, runs: int, steps: int):
        self.__runs = runs
        self.__steps = steps
        self.results = []
        # TODO: save params in json file
        self.params = [    
                        {
                            "epsilon": 0.1,
                            "bias": 0,
                            "step_size": 0
                        },
                        {
                            "epsilon": 0.01,
                            "bias": 0,
                            "step_size": 0
                        },
                        {
                            "epsilon": 0.0,
                            "bias": 0,
                            "step_size": 0
                        }
                    ]
        
        # TODO: save params in json file
        #params = [(0.1, 0, 0.1 ),
        #          (0.0, 5, 0.1)]

    def run(self):
        for triplet in self.params:
            results = BanditResults()
            for run_id in range(self.__runs):
                triplet["seed"] = run_id
                algorithm = AlgorithmCatalog.get_algorithm("IncrementalSimpleBandit", triplet)
                algorithm.run(self.__steps, results)
                results.save_current_run()
            self.results.append({"results": results, "params": triplet})