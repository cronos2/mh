import cProfile
import numpy as np

from algorithms import (
    ACEGeneticAlgorithm,
    ACSGeneticAlgorithm,
    BLXEGeneticAlgorithm,
    BLXSGeneticAlgorithm,
    LocalSearchAlgorithm,
    ReliefAlgorithm
)
from core import Classifier1NN
from utils import ArffReader, Result, ResultsCollector


def main():
    np.seterr(all='raise')  # always raise error, no runtime warnings
    np.random.seed(0)
    databases = ['sonar', 'spambase', 'wdbc']
    algorithms = [
        # ReliefAlgorithm,
        # LocalSearchAlgorithm,
        ACEGeneticAlgorithm,
        # ACSGeneticAlgorithm,
        # BLXEGeneticAlgorithm,
        # BLXSGeneticAlgorithm
    ]

    results = {}

    for alg in algorithms:
        collector = ResultsCollector()

        for db in databases:
            dataset = ArffReader.read(db)
            parts = dataset.generate_partitions()

            for i, part in enumerate(parts):
                name = '{db} - {i}'.format(db=db, i=i)
                res = Result(name=name)

                res.start_timer()

                learner = alg(part['train'])
                learner.train()
                learner.test(part['test'])

                res.end_timer()

                res.solution = learner.solution
                collector.append_result(res)

        results[alg.__name__] = collector

        # relief = ReliefAlgorithm(parts[0]['training'])
        # relief.train()
        # print(relief.w)
        # print 'Error RELIEF:', Classifier1NN(dataset).calculate_error(relief.w)

        # ls = LocalSearchAlgorithm(parts[0]['training'])
        # cProfile.runctx('ls.train()', globals={}, locals={'ls': ls})
        # print(ls.w)
        # print 'Error LS:', ls.classifier.calculate_error(ls.w)

        # ace = ACEGeneticAlgorithm(parts[0]['train'])
        # ace.train()
        # print(ace.solution.chromosome)

        # acs = ACSGeneticAlgorithm(parts[0]['train'])
        # acs.train()
        # print(acs.solution.chromosome)

        # classifier = Classifier1NN(parts[0]['train'])
        # print('ACE', classifier.calculate_error(w1))
        # print('ACS', classifier.calculate_error(w2))

    print results


if __name__ == '__main__':
    prof = cProfile.Profile()

    prof.enable()
    main()
    prof.disable()

    prof.print_stats(sort='time')
