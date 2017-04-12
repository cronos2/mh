import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform


class Solution(object):
    def __init__(self, chromosome, succ=None):
        self.chromosome = np.array(chromosome)
        self.succ = succ

    # comparison operators

    def __ge__(self, other):
        return self.succ.__ge__(other.succ)

    def __gt__(self, other):
        return self.succ.__gt__(other.succ)

    def __le__(self, other):
        return self.succ.__le__(other.succ)

    def __lt__(self, other):
        return self.succ.__lt__(other.succ)

    def __str__(self):
        return '{} ({})'.format(self.chromosome, self.succ)

    def __repr__(self):
        return str(self)

    @staticmethod
    def from_population(population):
        return np.array([Solution(chromosome) for chromosome in population])


class Classifier1NN(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def calculate_error(self, w):
        distances = squareform(pdist(self.dataset.observations, 'wminkowski', p=2, w=w))
        np.fill_diagonal(distances, np.nan)  # so we can use np.nanargmin
        closest = np.nanargmin(distances, axis=1)  # by rows (should be equivalent)
        error = np.mean(self.dataset.labels != self.dataset.labels[closest])

        return error

    def force_evaluation(self, solution):  # TODO: name - doesn't actually force (?)
        if solution.succ is None:
            chromosome = solution.chromosome
            error = self.calculate_error(chromosome)
            solution.succ = error

            return error
        else:
            return solution.succ

    def test_error(self, test_dataset, w):
        distances = cdist(
            test_dataset,
            self.dataset.observations,
            metric='wminkowski',
            p=2,
            w=w
        )

        closest = np.argmin(distances, axis=1)  # by rows
        guesses = self.dataset.labels[closest]
        error = np.mean(test_dataset.labels != guesses)

        return error


class Dataset(object):
    def __init__(self, dataset=None, labels=None, observations=None):
        if dataset is not None:
            self.set_labels([row[0] for row in dataset])
            self.set_observations([row[1:] for row in dataset])
        else:
            if labels is not None:
                self.set_labels(labels)

            if observations is not None:
                self.set_observations(observations)

    def set_labels(self, labels):
        self.labels = np.array(labels)

    def set_observations(self, observations):
        self.observations = np.array(observations)

    def generate_partitions(self, N=5):
        total_items = len(self.labels)
        permutations = [np.random.permutation(total_items) for _ in xrange(total_items - 1)]
        permutations = [np.arange(total_items)] + permutations

        partitions = [np.array_split(indices, 2) for indices in permutations]
        partitions = np.array([(
            {
                'train': Dataset(
                    labels=self.labels[partition[0]],
                    observations=self.observations[partition[0]]
                ),
                'test': Dataset(
                    labels=self.labels[partition[1]],
                    observations=self.observations[partition[1]]
                )
            },
            {
                'train': Dataset(
                    labels=self.labels[partition[1]],
                    observations=self.observations[partition[1]]
                ),
                'test': Dataset(
                    labels=self.labels[partition[0]],
                    observations=self.observations[partition[0]]
                )
        }) for partition in partitions]).flatten()

        return partitions
