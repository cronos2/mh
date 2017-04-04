import numpy as np

from algorithms import BaseAlgorithm
#from core import Classifier1NN


class BinaryTournamentSelectionOperator(object):
    @staticmethod
    def select(chromosomes, classifier):
        candidates = np.random.choice(chromosomes, 2)

        if classifier.calculate_error(candidates[0]) < classifier.calculate_error(candidates[1]):
            return candidates[0]
        else:
            return candidates[1]


class BlendAlphaCrossoverOperator(object):
    def __init__(self, probability, alpha):
        self.probability = probability
        self.alpha = alpha

    def crossover(self, parents):
        if np.random.rand() < self.probability:
            min_components = np.minimum(*parents)  # c_min
            max_components = np.maximum(*parents)  # c_max
            diff = max_components - min_components    # I = c_max - c_min

            baseline = (min_components - diff * self.alpha)  # c_min - I * a
            # c_max + I * a - (c_min - I * a) = c_max - c_min + 2 * I * a =
            amplitude = (1 + 2 * self.alpha) * diff  #  = (1 + 2a) * I
            offspring = np.random.rand(*parents.shape)  # h

            return baseline + offspring * amplitude
        else:
            return parents


class ArithmeticCrossover(object):
    def __init__(self, probability, alpha):
        self.probability = probability
        self.alpha = alpha

    def crossover(self, parents):
        if np.random.rand() < self.probability:
            return self.alpha * parents + (1 - self.alpha) * self.parents[::-1]
        else:
            return parents


class NormalMutationOperator(object):
    def __init__(self, probability, sigma):
        self.probability = probability
        self.sigma = sigma

    def mutate(self, chromosome):
        if np.random.rand() < self.probability:
            # chromosome = chromosome.chromosome  # get (ref to) actual chromosome from Solution object
            gene = np.random.choice(chromosome.shape[0])  # gene \in [0, length[
            chromosome[gene] += self.sigma * np.random.randn()


class GeneticAlgorithmMixin(object):
    def __init__(self, chromosomes, genes, selection, crossover, mutation):
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation

        self.population = [Solution(c) for c in np.random.rand(chromosomes, genes)]
        self.parents = []

    def train():
        # selection stage

        self.generate_parents()  # from Elitist/Stationary Mixin

        # crossover stage

        parents_chromosomes = [p.chromosome for p in self.parents]
        _iterator = iter(parents_chromosomes)
        couples = np.array(zip(_iterator, _iterator))  # s -> (s0, s1), (s2, s3), ...

        offspring = np.apply_over_axis(self.crossover.crossover, axis=1, arr=couples)
        offspring = offspring.reshape(-1, offspring.shape[-1])  # "flatten" array

        # mutation stage

        np.apply_along_axis(self.mutation.mutate, axis=1, arr=offspring)
        # ^ this should modify its argument chromosome in case of performance ^

        # replacement stage

        self.generate_population(offspring)  # from Elitist/Stationary Mixin


class ElitistMixin(object):
    def generate_parents(self):
        matches = [np.random.choice(self.population, 2) for _ in self.population]
        winners = [self.selection.select(match) for match in matches]
        self.parents = winners

    def generate_population(self, offspring):
        pass

class StationaryMixin(object):
    def generate_parents(self):
        matches = [np.random.choice(self.population, 2) for _ in xrange(2)]
        winners = [self.selection.select(match) for match in matches]
        self.parents = winners
