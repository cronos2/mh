import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform

from core import Classifier1NN
from genetic import (
    ArithmeticCrossoverOperator,
    BinaryTournamentSelectionOperator,
    ElitistMixin,
    GeneticAlgorithmMixin,
    NormalMutationOperator,
)


class BaseAlgorithm(object):
    pass


class LocalSearchAlgorithm(BaseAlgorithm):
    def __init__(self, dataset, max_evaluations=15000):
        self.classifier = Classifier1NN(dataset)
        self.N = len(dataset.observations[0])
        self.max_evaluations = max_evaluations
        self.max_neighbours = 20 * self.N

    def train(self):
        self.w = np.random.rand(self.N)
        self.best_solution_found = self.w
        self.best_solution_error = self.classifier.calculate_error(self.w)
        current_evaluations = 0
        current_neighbours = 0
        gene = 0

        while current_evaluations < self.max_evaluations and current_neighbours < self.max_neighbours:
            neighbour = self.w.copy()
            neighbour[gene % self.N] += np.random.randn()
            neighbour_error = self.classifier.calculate_error(neighbour)

            current_evaluations += 1
            gene += 1

            if neighbour_error < self.best_solution_error:
                self.w = neighbour
                self.best_solution_error = neighbour_error
                current_neighbours = 0
            else:
                current_neighbours += 1


class ReliefAlgorithm(BaseAlgorithm):
    def __init__(self, dataset):
        self.dataset = dataset
        self.split_datasets()

    def split_datasets(self):
        cond = self.dataset.labels == self.dataset.labels[0]
        self.A = self.dataset.observations[cond]
        self.B = self.dataset.observations[~cond]

        # We don't care about the actual distance but just need it to sort.
        # Therefore, we can use the squared euclidean metric which doesn't require
        # computing square roots and hence will be faster

        self.distances = {
            'AA': squareform(pdist(self.A, 'sqeuclidean')),
            'BB': squareform(pdist(self.B, 'sqeuclidean')),
            'AB': cdist(self.A, self.B, metric='sqeuclidean')
        }

        np.fill_diagonal(self.distances['AA'], np.nan)  # avoid considering oneself
        np.fill_diagonal(self.distances['BB'], np.nan)

    def train(self):
        closest_friends = {
            'A': np.nanargmin(self.distances['AA'], axis=1),  # by rows
            'B': np.nanargmin(self.distances['BB'], axis=1)
        }

        closest_enemies = {
            'AB': np.argmin(self.distances['AB'], axis=1),
            'BA': np.argmin(self.distances['AB'], axis=0)  # by cols (!)
        }

        self.w = np.sum(np.abs(
            self.A - self.B[closest_enemies['AB']]
        ), axis=0) + np.sum(np.abs(
            self.B - self.A[closest_enemies['BA']]
        ), axis=0) - np.sum(np.abs(
            self.A - self.A[closest_friends['A']]
        ), axis=0) - np.sum(np.abs(
            self.B - self.B[closest_friends['B']]
        ), axis=0)

        # normalize w

        self.w[self.w < 0] = 0  # truncate negative values
        self.w[self.w > 0] /= self.w.max()  # normalize to [0, 1]
        # ^ this will NEVER divide by 0 ^


class ACEGeneticAlgorithm(BaseAlgorithm, GeneticAlgorithmMixin, ElitistMixin):
    def __init__(self, dataset):
        super(ACEGeneticAlgorithm, self).__init__(
            n_chromosomes=30,
            n_genes=len(dataset.observations[0]),
            max_evaluations=15000
        )

        self.classifier = Classifier1NN(dataset)
        self.selection = BinaryTournamentSelectionOperator()
        self.crossover = ArithmeticCrossoverOperator(probability=0.7, alpha=0.3)
        self.mutation = NormalMutationOperator(probability=0.001, sigma=0.3)
