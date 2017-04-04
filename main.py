import arff
import cProfile
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform

from algorithms import LocalSearchAlgorithm, ReliefAlgorithm

def memoize(function):  # TODO: manage memory
    '''
    This will create a cache dictionary for every bound method of the class
    When does it free memory?
    '''
    cache = {}
    def decorated_function(*args):
        if args in cache:
            print 'cached'
            return cache[args]
        else:
            print 'not cached'
            val = function(*args)
            cache[args] = val
            return val
    return decorated_function


class ArffReader(object):
    @staticmethod
    def read(path):
        common = {
            'sonar': 'datasets/sonar.arff',
            'spambase': 'datasets/spambase-460.arff',
            'wdbc': 'datasets/wdbc.arff'
        }

        f = open(common.get(path, path))
        data = arff.loads(f)

        return Dataset(data['data'])


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
        partitions = [{
            'training': Dataset(
                labels=self.labels[partition[0]],
                observations=self.observations[partition[0]]
            ),
            'test': Dataset(
                labels=self.labels[partition[1]],
                observations=self.observations[partition[1]]
            )
        } for partition in partitions]

        return partitions


class Classifier1NN(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def calculate_error(self, w):
        distances = squareform(pdist(self.dataset.observations, 'wminkowski', p=2, w=w))
        np.fill_diagonal(distances, np.nan)  # so we can use np.nanargmin
        closest = np.nanargmin(distances, axis=1)  # by rows (should be equivalent)
        error = np.mean(self.dataset.labels != self.dataset.labels[closest])

        return error


def main():
    np.random.seed(0)
    dataset = ArffReader.read('spambase')
    # dataset = Dataset(np.array([[0, 0, 0], [0, 1, 0], [1, 0, 2], [1, 1, 2]]))
    parts = dataset.generate_partitions()

    relief = ReliefAlgorithm(parts[0]['training'])
    relief.train()
    print(relief.w)
    print 'Error RELIEF:', Classifier1NN(dataset).calculate_error(relief.w)

    ls = LocalSearchAlgorithm(parts[0]['training'])
    cProfile.runctx('ls.train()', globals={}, locals={'ls': ls})
    print(ls.w)
    print 'Error LS:', ls.classifier.calculate_error(ls.w)



if __name__ == '__main__':
    main()
