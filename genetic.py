import numpy as np

from core import Solution


class BinaryTournamentSelectionOperator(object):
    def select(self, chromosomes):
        '''
        This expects TWO chromosomes. They should be evaluated.
        Warning: will work otherwise, but my lead to logic errors
        '''

        return np.max(chromosomes)  # max score


class BlendAlphaCrossoverOperator(object):
    def __init__(self, probability, alpha):
        self.probability = probability
        self.alpha = alpha

    def crossover(self, parents):
        if np.random.rand() < self.probability:
            parents = np.array([parent.w for parent in parents])  # unpack Solution
            min_components = np.minimum(*parents)  # c_min
            max_components = np.maximum(*parents)  # c_max
            diff = max_components - min_components    # I = c_max - c_min

            baseline = (min_components - diff * self.alpha)  # c_min - I * a
            # c_max + I * a - (c_min - I * a) = c_max - c_min + 2 * I * a =
            amplitude = (1 + 2 * self.alpha) * diff  #  = (1 + 2a) * I
            offspring = np.random.rand(*parents.shape)  # h

            return Solution.from_population(baseline + offspring * amplitude)
        else:
            return parents


class ArithmeticCrossoverOperator(object):
    def __init__(self, probability, alpha):
        self.probability = probability
        self.alpha = alpha

    def crossover(self, parents):
        if np.random.rand() < self.probability:
            parents = np.array([parent.w for parent in parents])  # unpack Solution
            offspring = self.alpha * parents + (1 - self.alpha) * parents[::-1]
            return Solution.from_population(offspring)
        else:
            return parents


class NormalMutationOperator(object):
    def __init__(self, probability, sigma):
        self.probability = probability
        self.sigma = sigma

    def mutate(self, chromosome):
        chromosome = chromosome.w.copy()  # unpack Solution
        r = np.random.rand(*chromosome.shape)
        mask = r < self.probability
        chromosome[mask] += self.sigma * np.random.randn(np.sum(mask))

        result = Solution(chromosome)
        result.normalize()
        return result


class GeneticAlgorithmMixin(object):
    def __init__(self, n_chromosomes, n_genes, max_evaluations):
        self.population = np.array([Solution(c) for c in np.random.rand(n_chromosomes, n_genes)])
        self.n_chromosomes = n_chromosomes
        self.n_genes = n_genes
        self.parents = np.array([])

        self.max_evaluations = max_evaluations
        self.current_evaluations = 0

    def train(self, max_generations=0):
        for individual in self.population:
            self.classifier.evaluate_solution(individual)

        self.current_evaluations = self.n_chromosomes  # == len(self.population)
        current_generation = 0

        # if max_generations is 0 then it doesn't set up a limit
        while (self.current_evaluations < self.max_evaluations and
                (current_generation < max_generations or not max_generations)):

            current_generation += 1

            # selection stage

            self.generate_parents()  # from Elitist/Stationary Mixin

            # crossover stage

            _iterator = iter(self.parents)
            couples = np.array(zip(_iterator, _iterator))  # s -> (s0, s1), (s2, s3), ...

            offspring = np.apply_along_axis(self.crossover.crossover, axis=1, arr=couples)
            offspring = offspring.reshape(-1)  # "flatten" array

            # mutation stage

            offspring = np.array([self.mutation.mutate(child) for child in offspring])

            # replacement stage

            self.generate_population(offspring)  # from Elitist/Stationary Mixin
            # ^ this updates self.current_evaluations ^

        # end of training

        self.solution = np.max(self.population)  # max score

        return self.current_evaluations


class ElitistMixin(object):
    def generate_parents(self):
        matches = [np.random.choice(self.population, 2, replace=False) for _ in self.population]
        winners = np.array([self.selection.select(match) for match in matches])
        self.parents = winners

    def generate_population(self, offspring):
        # force all evaluations

        for child in offspring:
            self.classifier.evaluate_solution(child)

        self.current_evaluations += offspring.shape[0]

        # "replace" worst child for best parent

        best = np.argmax(self.population)  # max score

        if self.population[best] not in offspring:  # doesn't survive
            worst = np.argmin(offspring)  # min score
            offspring[worst] = self.population[best]

        # actual replacement

        self.population = offspring
        self.parents = np.array([])


class StationaryMixin(object):
    def generate_parents(self):
        matches = [np.random.choice(self.population, 2, replace=False) for _ in xrange(2)]
        winners = np.array([self.selection.select(match) for match in matches])
        self.parents = winners

    def generate_population(self, offspring):
        # force all evaluations

        for child in offspring:
            self.classifier.evaluate_solution(child)

        self.current_evaluations += 2

        # replace worst parents for best children

        self.population.partition(2)  # two worst (smaller score)
        contestants = np.concatenate((self.population[:2], offspring))
        selected = np.partition(contestants, -1)[-2:]  # two best

        self.population[:2] = selected
        self.parents = np.array([])
