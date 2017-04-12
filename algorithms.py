import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform

from core import Classifier1NN, Solution
from genetic import (
    ArithmeticCrossoverOperator,
    BinaryTournamentSelectionOperator,
    ElitistMixin,
    StationaryMixin,
    GeneticAlgorithmMixin,
    NormalMutationOperator,
)


class BaseAlgorithm(object):
    def test(self, test_dataset):
        return self.classifier.test_error(test_dataset, self.solution.w)


class LocalSearchAlgorithm(BaseAlgorithm):
    def __init__(self, dataset, max_evaluations=15000):
        self.classifier = Classifier1NN(dataset)
        self.max_evaluations = max_evaluations
        self.n_features = dataset.observations.shape[1]  # number of columns
        self.max_neighbours = 20 * self.n_features

    def train(self):
        self.solution = Solution(np.random.rand(self.n_features))
        self.classifier.evaluate_solution(self.solution)

        current_evaluations = 0
        current_neighbours = 0
        gene = 0

        while current_evaluations < self.max_evaluations and current_neighbours < self.max_neighbours:
            neighbour = Solution(self.solution.w.copy())
            neighbour.w[gene % self.n_features] += np.random.randn()
            neighbour.normalize()
            self.classifier.evaluate_solution(neighbour)

            current_evaluations += 1
            gene += 1

            if neighbour.error < self.solution.error:
                self.solution = neighbour
                current_neighbours = 0
            else:
                current_neighbours += 1


class ReliefAlgorithm(BaseAlgorithm):
    def __init__(self, dataset):
        self.classifier = Classifier1NN(dataset)  # adheres to BaseAlgorithm if
        self.dataset = dataset
        self.split_datasets()

    def split_datasets(self):
        cond = self.dataset.labels == self.dataset.labels[0]
        self.A = self.dataset.observations[cond]
        self.B = self.dataset.observations[~cond]

        try:
            assert self.A.shape[0] != 0
            assert self.B.shape[0] != 0
        except AssertionError:
            print 'Not enough representatives of either class for the RELIEF algorithm'
            raise  # re-raise exception

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

        self.solution = Solution(self.w)  # adheres to BaseAlgorithm interface
        self.solution.normalize()
        self.classifier.evaluate_solution(self.solution)


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


class ACSGeneticAlgorithm(BaseAlgorithm, GeneticAlgorithmMixin, StationaryMixin):
    def __init__(self, dataset):
        super(ACSGeneticAlgorithm, self).__init__(
            n_chromosomes=30,
            n_genes=len(dataset.observations[0]),
            max_evaluations=15000
        )

        self.classifier = Classifier1NN(dataset)
        self.selection = BinaryTournamentSelectionOperator()
        self.crossover = ArithmeticCrossoverOperator(probability=0.7, alpha=0.3)
        self.mutation = NormalMutationOperator(probability=0.001, sigma=0.3)


class BLXEGeneticAlgorithm(BaseAlgorithm, GeneticAlgorithmMixin, ElitistMixin):
    def __init__(self, dataset):
        super(BLXEGeneticAlgorithm, self).__init__(
            n_chromosomes=30,
            n_genes=len(dataset.observations[0]),
            max_evaluations=15000
        )

        self.classifier = Classifier1NN(dataset)
        self.selection = BinaryTournamentSelectionOperator()
        self.crossover = BlendAlphaCrossoverOperator(probability=0.7, alpha=0.3)
        self.mutation = NormalMutationOperator(probability=0.001, sigma=0.3)


class BLXSGeneticAlgorithm(BaseAlgorithm, GeneticAlgorithmMixin, StationaryMixin):
    def __init__(self, dataset):
        super(BLXSGeneticAlgorithm, self).__init__(
            n_chromosomes=30,
            n_genes=len(dataset.observations[0]),
            max_evaluations=15000
        )

        self.classifier = Classifier1NN(dataset)
        self.selection = BinaryTournamentSelectionOperator()
        self.crossover = BlendAlphaCrossoverOperator(probability=0.7, alpha=0.3)
        self.mutation = NormalMutationOperator(probability=0.001, sigma=0.3)
