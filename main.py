# coding: utf-8

import argparse
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

    parser = argparse.ArgumentParser(description='Ejecuta los experimentos de la primera práctica de Metaheurísticas')
    parser.add_argument(
        '-i, --interactive',
        action='store_true',
        dest='interactive'
    )
    parser.add_argument(
        '-o, --output',
        dest='output_filename',
        default=None
    )
    args = parser.parse_args()

    databases = ['sonar', 'spambase', 'wdbc']
    algorithms = [
        ReliefAlgorithm,
        LocalSearchAlgorithm,
        ACEGeneticAlgorithm,
        ACSGeneticAlgorithm,
        BLXEGeneticAlgorithm,
        BLXSGeneticAlgorithm
    ]

    results = {}

    for alg in algorithms:
        collector = ResultsCollector()

        for db in databases:
            dataset = ArffReader.read(db)
            partitions = dataset.generate_partitions()

            for i, partition in enumerate(partitions):
                name = '{db} - {i}'.format(db=db, i=i)
                res = Result(name=name)
                res.indices = {
                    name: array.tolist()  # cast np arrays to normal lists (JSON)
                    for name, array in partition.indices.iteritems()
                }

                res.start_timer()

                learner = alg(partition.training_set)
                learner.train()

                res.end_timer()

                # calculate errors

                train_error = learner.classifier.evaluate_solution(
                    learner.solution
                )
                test_error = learner.test(partition.testing_set)

                # set up Result

                res.solution = learner.solution.w.tolist()  # cast np array to list
                res.train_error = train_error
                res.test_error = test_error

                if args.interactive:
                    print(res)

                collector.append_result(res)

        results[alg.__name__] = collector

    if args.output_filename is None:
        print results
    else:
        f = open(args.output_filename, 'wb')
        f.write(str(results))
        f.close()


if __name__ == '__main__':
    # prof = cProfile.Profile()

    # prof.enable()
    # main()
    # prof.disable()

    # prof.print_stats(sort='time')

    main()
