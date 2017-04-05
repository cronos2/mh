import numpy as np


class BinaryTournamentSelectionOperator(object):
    def select(self, chromosomes):
        # this expects all candidates to be evaluated (!)

        candidates = np.random.choice(chromosomes, 2)
        return min(candidates)  # less error


class BlendAlphaCrossoverOperator(object):
    def __init__(self, probability, alpha):
        self.probability = probability
        self.alpha = alpha

    def crossover(self, parents):
        if np.random.rand() < self.probability:
            parents = np.array([parent.chromosome for parent in parents])  # unpack Solution
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
            parents = [parent.chromosome for parent in parents]  # unpack Solution
            offspring = self.alpha * parents + (1 - self.alpha) * parents[::-1]
            return Solution.from_population(offspring)
        else:
            return parents


class NormalMutationOperator(object):
    def __init__(self, probability, sigma):
        self.probability = probability
        self.sigma = sigma

    def mutate(self, chromosome):  # TODO: copy chromosome array
        chromosome = chromosome.chromosome.copy()  # unpack Solution
        r = np.random.rand(*chromosome.shape)
        mask = r < self.probability
        chromosome[mask] += self.sigma * np.random.randn(np.sum(mask))

        return Solution(chromosome)

        # if np.random.rand() < self.probability:  # TODO: this should act by GENE (!)
        #     gene = np.random.choice(chromosome.shape[0])  # gene \in [0, length[
        #     chromosome[gene] += self.sigma * np.random.randn()


class GeneticAlgorithmMixin(object):
    def __init__(self, n_chromosomes, n_genes, max_evaluations):
        self.population = [Solution(c) for c in np.random.rand(chromosomes, genes)]
        self.n_chromosomes = n_chromosomes
        self.n_genes = n_genes
        self.parents = []
        self.max_evaluations = max_evaluations
        self.current_evaluations = 0

    def train():
        while self.current_evaluations < self.max_evaluations:
            # selection stage

            self.generate_parents()  # from Elitist/Stationary Mixin

            # crossover stage

            parents_chromosomes = [p.chromosome for p in self.parents]
            _iterator = iter(self.parents)
            couples = np.array(zip(_iterator, _iterator))  # s -> (s0, s1), (s2, s3), ...

            offspring = np.apply_along_axis(self.crossover.crossover, axis=1, arr=couples)
            offspring = offspring.reshape(-1, offspring.shape[-1])  # "flatten" array

            # mutation stage

            np.apply_along_axis(self.mutation.mutate, axis=1, arr=offspring)
            # ^ this should modify its argument chromosome in case of performance ^
            # ^ FALSE ^

            # replacement stage

            self.generate_population(offspring)  # from Elitist/Stationary Mixin
            # ^ this updates self.current_evaluations ^

        # end of training

        self.solution = np.min(self.population)  # less error


class ElitistMixin(object):
    def generate_parents(self):
        matches = [np.random.choice(self.population, 2) for _ in self.population]
        winners = [self.selection.select(match) for match in matches]
        self.parents = winners

    def generate_population(self, offspring):
        # force all evaluations

        for individual in self.population:
            self.classifier.force_evaluation(individual)

        for child in offspring:
            self.classifier.force_evaluation(child)

        self.current_evaluations += self.n_chromosomes + len(offspring)

        # "replace" worst child for best parent

        worst = np.argmax(offspring)  # max error
        best = np.argmin(self.population)  # min error

        if self.population[best] not in offspring[worst]:  # doesn't survive
            offspring[worst] = self.population[best]

        # actual replacement

        self.population = offspring
        self.parents = []


class StationaryMixin(object):
    def generate_parents(self):
        matches = [np.random.choice(self.population, 2) for _ in xrange(2)]
        winners = [self.selection.select(match) for match in matches]
        self.parents = winners

    def generate_population(self, offspring):
        # force all evaluations

        for individual in self.population:
            self.classifier.force_evaluation(individual)

        for child in offspring:
            self.classifier.force_evaluation(child)

        self.current_evaluations += self.n_chromosomes + len(offspring)  # TODO: improve upper bound

        # replace worst parents for best children

        self.population.partition(-1)  # two worst (greater ones)
        offspring.partition(1)  # two best
        contestants = np.stack((self.population[-1:], offspring[:1]))
        selected = np.partition(contestants, 1)[:1]

        self.population[-1:] = selected
        self.parents = []


'''
TODO: evaluation counting
'''
