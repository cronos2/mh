import cProfile
import numpy as np

from algorithms import (
    ACEGeneticAlgorithm,
    LocalSearchAlgorithm,
    ReliefAlgorithm
)
from core import Classifier1NN
from utils import ArffReader


def main():
    np.random.seed(0)
    dataset = ArffReader.read('spambase')
    # dataset = Dataset(np.array([[0, 0, 0], [0, 1, 0], [1, 0, 2], [1, 1, 2]]))
    parts = dataset.generate_partitions()

    # relief = ReliefAlgorithm(parts[0]['training'])
    # relief.train()
    # print(relief.w)
    # print 'Error RELIEF:', Classifier1NN(dataset).calculate_error(relief.w)

    # ls = LocalSearchAlgorithm(parts[0]['training'])
    # cProfile.runctx('ls.train()', globals={}, locals={'ls': ls})
    # print(ls.w)
    # print 'Error LS:', ls.classifier.calculate_error(ls.w)

    ace = ACEGeneticAlgorithm(parts[0]['training'])
    ace.train()
    print(ace.solution.chromosome)

if __name__ == '__main__':
    main()
