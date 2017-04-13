import numpy as np


class MemeticAlgorithmMixin(object):
    def __init__(self, max_evaluations):
        self.max_evaluations = max_evaluations
        self.current_evaluations = 0

    def _reset_ga(self):
        self.ga.current_evaluations = 0
        self.ga.max_evaluations = min(
            self.ga.max_evaluations,
            self.max_evaluations - self.current_evaluations
        )  # maybe there are not enough available evaluations left

    def _reset_exploiter(self):
        self.exploiter.max_evaluations = min(
            self.exploiter.max_evaluations,
            self.max_evaluations - self.current_evaluations
        )  # maybe there are not enough available evaluations left


    def train(self):
        while self.current_evaluations < self.max_evaluations:
            self._reset_ga()

            ga_evaluations = self.ga.train(max_generations=self.ga_generations)
            self.current_evaluations += ga_evaluations

            self.exploit()  # this increments the evaluations counter


        self.solution = np.min(self.ga.population)  # best one
