import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform


class Solution(object):
    def __init__(self, w, error=None, score=None):
        self.w = np.array(w)
        self.error = error
        self.score = score

    def normalize(self):
        self.w[self.w < 0] = 0  # truncate negative values
        if self.w.max() > 1:
            self.w[self.w > 0] /= self.w.max()  # normalize to [0, 1]
            # ^ this will NEVER divide by 0 ^

    # comparison operators

    def __ge__(self, other):
        return self.score.__ge__(other.score)

    def __gt__(self, other):
        return self.score.__gt__(other.score)

    def __le__(self, other):
        return self.score.__le__(other.score)

    def __lt__(self, other):
        return self.score.__lt__(other.score)

    def __str__(self):
        return '{} ({})'.format(self.w, self.score)

    def __repr__(self):
        return str(self)

    @property
    def redux(self):
        return np.mean(self.w < 0.1)  # features "unused" / total features

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
        if solution.score is None:
            weights = solution.w
            error = self.train_error(weights)
            solution.error = error

            solution.score = np.mean([1 - solution.error, solution.redux])

        return solution.score

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

    def test_solution(self, test_dataset, solution):
        error = self.test_error(test_dataset, solution.w)

        return np.mean([1 - error, solution.redux])


class Partition(object):
    def __init__(self, dataset, indices):
        self.training_set = dataset.from_indices(indices[0])
        self.testing_set = dataset.from_indices(indices[1])
        self.indices = dict(zip(('training', 'testing'), indices))


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
        self.normalize_observations()

    def normalize_observations(self):
        features_max = self.observations.max(axis=0)
        features_min = self.observations.min(axis=0)
        diff = (features_max - features_min)

        self.observations = self.observations - features_min  # baseline

        try:
            self.observations /= diff  # -> [0, 1]
        except FloatingPointError:  # dividing by zero
            # there are some features sharing the same value in every observation
            mask = (diff == 0)

            self.observations[:, ~mask] /= diff[~mask]  # normalize the non-zero
            self.observations[:, mask] = 0 # this IS necessary (otherwise nan)

    def from_indices(self, indices):
        # v this IS necessary (otherwise renormalization) v
        d = Dataset()  # set values directly to avoid rechecking
        d.labels = self.labels[indices]
        d.observations = self.observations[indices]

        return d

    def generate_partitions(self, N=5):
        total_items = self.labels.shape[0]
        permutations = [np.random.permutation(total_items) for _ in xrange(N)]
        # permutations = [np.arange(total_items)] + permutations

        partitions_indices = [
            np.array_split(indices, 2)
        for indices in permutations]

        partitions = np.array([
            (Partition(self, indices), Partition(self, indices[::-1]))
            for indices in partitions_indices]).flatten()

        return partitions
    def kfoldcv(self, K=5):
        total_items = np.size(self.labels)
        shuffle = np.random.permutation(total_items)

        partitions_indices = np.array_split(shuffle, K)

        partitions = [Partition(self, [
            np.delete(partitions_indices, i, 0).flatten(),
            partitions_indices[i]
        ])
                      for i in xrange(K)]

        return partitions



class Conditions(object):
