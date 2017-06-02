from collections import Counter
import math
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform

from core import Classifier1NN, Solution
from genetic import (
    ArithmeticCrossoverOperator,
    BinaryTournamentSelectionOperator,
    BlendAlphaCrossoverOperator,
    ElitistMixin,
    StationaryMixin,
    GeneticAlgorithmMixin,
    NormalMutationOperator,
)
from memetic import MemeticAlgorithmMixin


class BaseAlgorithm(object):
    def test(self, test_dataset):
        return self.classifier.test_solution(test_dataset, self.solution)


class LocalSearchAlgorithm(BaseAlgorithm):
    def __init__(self, dataset, max_evaluations=15000, attempts_per_gene=20):
        self.classifier = Classifier1NN(dataset)
        self.max_evaluations = max_evaluations
        self.n_features = dataset.observations.shape[1]  # number of columns
        self.max_neighbours = attempts_per_gene * self.n_features
        self.solution = Solution(np.random.rand(self.n_features))

    def train(self):
        self.classifier.evaluate_solution(self.solution)

        current_evaluations = 0
        current_neighbours = 0
        gene = 0

        while (current_evaluations < self.max_evaluations and
               current_neighbours < self.max_neighbours):
            neighbour = Solution(self.solution.w.copy())
            neighbour.w[gene % self.n_features] += np.random.randn()
            neighbour.normalize()
            self.classifier.evaluate_solution(neighbour)

            current_evaluations += 1
            gene += 1

            if neighbour > self.solution:  # more score
                self.solution = neighbour
                current_neighbours = 0
            else:
                current_neighbours += 1

        return current_evaluations


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

        w = np.sum(np.abs(
            self.A - self.B[closest_enemies['AB']]
        ), axis=0) + np.sum(np.abs(
            self.B - self.A[closest_enemies['BA']]
        ), axis=0) - np.sum(np.abs(
            self.A - self.A[closest_friends['A']]
        ), axis=0) - np.sum(np.abs(
            self.B - self.B[closest_friends['B']]
        ), axis=0)

        self.solution = Solution(w)  # adheres to BaseAlgorithm interface
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
        self.crossover = ArithmeticCrossoverOperator(probability=1, alpha=0.3)
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
            n_genes=dataset.observations.shape[1],
            max_evaluations=15000
        )

        self.classifier = Classifier1NN(dataset)
        self.selection = BinaryTournamentSelectionOperator()
        self.crossover = BlendAlphaCrossoverOperator(probability=1, alpha=0.3)
        self.mutation = NormalMutationOperator(probability=0.001, sigma=0.3)


class MemeticAdaptedGA(ACEGeneticAlgorithm):
    def __init__(self, dataset):
        super(MemeticAdaptedGA, self).__init__(dataset)
        GeneticAlgorithmMixin.__init__(
            self,
            n_chromosomes=10,
            n_genes=dataset.observations.shape[1],
            max_evaluations=1500
        )  # set GA parameters accordingly


class BaseMemeticAlgorithm(BaseAlgorithm, MemeticAlgorithmMixin):
    def __init__(self, dataset):
        super(BaseMemeticAlgorithm, self).__init__(
            max_evaluations=15000
        )

        self.ga = MemeticAdaptedGA(dataset)
        self.ga_generations = 10
        self.exploiter = LocalSearchAlgorithm(
            dataset,
            max_evaluations=2 * dataset.observations.shape[1],
            attempts_per_gene=2
        )

        self.classifier = self.ga.classifier  # can use the same one


class MemeticAlgorithmA(BaseMemeticAlgorithm):
    def exploit(self):  # every individual gets exploited
        for i, agent in enumerate(self.ga.population):
            self._reset_exploiter()
            self.exploiter.solution = agent
            self.current_evaluations += self.exploiter.train()  # returns used evals
            print 'Current evaluations', self.current_evaluations

            self.ga.population[i] = self.exploiter.solution  # replace agent


class MemeticAlgorithmB(BaseMemeticAlgorithm):
    def exploit(self):  # just a (random) 10% of the population
        n_chromosomes = self.ga.population.shape[0]
        indices = np.random.choice(n_chromosomes, n_chromosomes / 10, replace=False)

        for index in indices:
            self._reset_exploiter()
            agent = self.ga.population[index]
            self.exploiter.solution = agent
            self.current_evaluations += self.exploiter.train()  # returns used evals

            self.ga.population[index] = self.exploiter.solution  # replace agent


class MemeticAlgorithmC(BaseMemeticAlgorithm):
    def exploit(self):  # just the best 10% of the population
        n_chromosomes = self.ga.population.shape[0]
        tenth = n_chromosomes / 10
        self.ga.population.partition(tenth - 1)  # -1 to get index
        # ^ this puts the <tenth> best agents in the front ^

        for index in xrange(tenth):
            self._reset_exploiter()
            agent = self.ga.population[index]
            self.exploiter.solution = agent
            self.current_evaluations += self.exploiter.train()  # returns used evals

            self.ga.population[index] = self.exploiter.solution  # replace agent


class SimulatedAnnealingAlgorithm(BaseAlgorithm):
    def __init__(self, dataset, max_evaluations=15000, mu=0.3, phi=0.3):
        self.classifier = Classifier1NN(dataset)
        self.n_features = dataset.observations.shape[1]  # number of columns
        self.solution = Solution(np.random.rand(self.n_features))
        self.mu = mu
        self.phi = phi

        self.limits = {
            'evaluations': max_evaluations,
            'neighbours': 10 * self.n_features,
            'successes': self.n_features
        }

        self.state = Counter()

    def train(self):
        ## SETUP ##

        self.state.clear()
        self.best_solution = self.solution
        cost = self.classifier.evaluate_solution(self.solution)
        self.state.update(['evaluations'])
        self.temperature = - self.mu * cost / math.log(self.phi)
        self.final_temperature = 1e-3

        # v final temperature must be lower! v
        assert self.final_temperature < self.temperature

        M = self.limits['evaluations'] / self.limits['neighbours']
        self.beta = (self.temperature - self.final_temperature) / (M * self.temperature * self.final_temperature)

        self.state.update(['successes'])  # hack first iteration

        while (self.state['evaluations'] <= self.limits['evaluations'] and
               self.state['successes'] > 0):

            self.state['neighbours'] = 0
            self.state['successes'] = 0

            while (self.state['evaluations'] <= self.limits['evaluations'] and
                   self.state['neighbours'] <= self.limits['neighbours'] and
                   self.state['successes'] <= self.limits['successes']):
                index = np.random.randint(self.n_features)

                neighbour = Solution(self.solution.w.copy())
                self.state.update(['neighbours'])
                feature = np.random.randint(self.n_features)
                neighbour.w[feature] += np.random.randn()
                neighbour.normalize()
                self.classifier.evaluate_solution(neighbour)
                self.state.update(['evaluations'])

                delta_score = neighbour.score - self.solution.score

                if (delta_score > 0 or
                    np.random.rand() <= math.exp(delta_score / self.temperature)):
                    self.solution = neighbour
                    self.state.update(['successes'])

                    if self.solution > self.best_solution:  # more score
                        self.best_solution = self.solution

            self._update_temperature()

        if self.state['successes'] <= 0:
            print self.state

        return self.best_solution

    def _update_temperature(self):
        self.temperature = self.temperature / (1 + self.beta * self.temperature)


class IteratedLocalSearchAlgorithm(BaseAlgorithm):
    def __init__(self, dataset, max_evaluations=15000, n_exploits=15, mutation_factor=0.1, sigma=0.4):
        self.classifier = Classifier1NN(dataset)
        self.max_evaluations = max_evaluations
        self.n_features = dataset.observations.shape[1]  # number of columns
        self.n_exploits = n_exploits
        self.solution = Solution(np.random.rand(self.n_features))
        self.t = np.round(self.n_features * mutation_factor).astype(int)  # force cast
        self.sigma = sigma

    def train(self):
        self.classifier.evaluate_solution(self.solution)

        for _ in xrange(self.n_exploits):
            ls = LocalSearchAlgorithm(
                dataset=self.classifier.dataset,
                max_evaluations=self.max_evaluations / self.n_exploits
            )

            ls.solution = self._mutate(self.solution)  # override default random solution
            ls.train()

            if ls.solution > self.solution:  # more score
                self.solution = ls.solution

        return self.solution

    def _mutate(self, solution):
        indices = np.random.choice(self.n_features, self.t, replace=False)

        mutated = Solution(solution.w.copy())
        mutated.w[indices] += np.random.randn(self.t) * self.sigma
        mutated.normalize()

        return mutated


class DifferentialEvolutionMixin(object):
    def __init__(self, n_population, n_genes, max_evaluations, CR=0.5, F=0.5):
        self.population = np.array([Solution(c) for c in np.random.rand(n_population, n_genes)])
        self.n_population = n_population
        self.n_genes = n_genes
        self.max_evaluations = max_evaluations
        self.current_evaluations = 0
        self.CR = CR
        self.F = F

    def train(self):
        for pop in self.population:
            self.classifier.evaluate_solution(pop)

        self.current_evaluations = self.n_population

        while self.current_evaluations < self.max_evaluations:
            parents = self.generate_parents()  # actually indices
            offspring = np.array([Solution(pop.w.copy()) for pop in self.population])

            for i, child in enumerate(offspring):
                self.crossover(child, self.population[parents[i]])

            self.current_evaluations += np.size(offspring)
            self.population = np.maximum(self.population, offspring)

        self.solution = np.max(self.population)  # best score


class DifferentialEvolutionRandomAlgorithm(BaseAlgorithm, DifferentialEvolutionMixin):
    def __init__(self, dataset):
        super(DifferentialEvolutionRandomAlgorithm, self).__init__(
            n_population=50,
            n_genes=dataset.observations.shape[1],  # number of columns
            max_evaluations=15000,
            CR=0.5,
            F=0.5
        )

        self.classifier = Classifier1NN(dataset)

    def generate_parents(self):
        return np.array([np.random.choice(
            self.n_population,
            3,
            replace=False
        ) for _ in self.population])  # 3 parents for every individual

    def crossover(self, child, parents):
        crossover_mask = np.random.rand(self.n_genes) < self.CR

        child.w[crossover_mask] = parents[0].w[crossover_mask] + self.F * (parents[1].w[crossover_mask] - parents[2].w[crossover_mask])
        child.normalize()

        self.classifier.evaluate_solution(child)


class DifferentialEvolutionCTBAlgorithm(BaseAlgorithm, DifferentialEvolutionMixin):
    def __init__(self, dataset):
        super(DifferentialEvolutionCTBAlgorithm, self).__init__(
            n_population=50,
            n_genes=dataset.observations.shape[1],  # number of columns
            max_evaluations=15000,
            CR=0.5,
            F=0.5
        )

        self.classifier = Classifier1NN(dataset)

    def generate_parents(self):
        self.current_best = np.max(self.population)

        return np.array([np.random.choice(
            self.n_population,
            2,
            replace=False
        ) for _ in self.population])  # 2 parents for every individual

    def crossover(self, child, parents):
        crossover_mask = np.random.rand(self.n_genes) < self.CR

        child.w[crossover_mask] += self.F * (
            self.current_best.w[crossover_mask] -
            child.w[crossover_mask] +
            parents[0].w[crossover_mask] -
            parents[1].w[crossover_mask]
        )
        child.normalize()

        self.classifier.evaluate_solution(child)
