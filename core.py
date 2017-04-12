import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform


class Solution(object):
    def __init__(self, w, error=None):
        self.w = np.array(w)
        self.error = error

    # comparison operators

    def __ge__(self, other):
        return self.error.__ge__(other.error)

    def __gt__(self, other):
        return self.error.__gt__(other.error)

    def __le__(self, other):
        return self.error.__le__(other.error)

    def __lt__(self, other):
        return self.error.__lt__(other.error)

    def __str__(self):
        return '{} ({})'.format(self.w, self.error)

    def __repr__(self):
        return str(self)

    @property
    def succ(self):
        return 1 - self.error

    @staticmethod
    def from_population(population):
        return np.array([Solution(w) for w in population])


class Classifier1NN(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def train_error(self, w):
        distances = squareform(
            pdist(
                np.sqrt(w) * self.dataset.observations,
                metric='sqeuclidean',
            )
        )

        np.fill_diagonal(distances, np.nan)  # so we can use np.nanargmin
        closest = np.nanargmin(distances, axis=1)  # by rows (should be equivalent)
        error = np.mean(self.dataset.labels != self.dataset.labels[closest])

        return error

    def evaluate_solution(self, solution):
        if solution.error is None:
            weights = solution.w
            error = self.train_error(weights)
            solution.error = error

            return error
        else:
            return solution.error

    def test_error(self, test_dataset, w):
        distances = cdist(
            np.sqrt(w) * test_dataset.observations,
            np.sqrt(w) * self.dataset.observations,
            metric='sqeuclidean',
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
        permutations = [np.random.permutation(total_items) for _ in xrange(N)]
        # permutations = [np.arange(total_items)] + permutations

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
