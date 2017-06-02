# coding: utf-8

import argparse
import cProfile
import json
import numpy as np

from algorithms import (
    ACEGeneticAlgorithm,
    ACSGeneticAlgorithm,
    BLXEGeneticAlgorithm,
    BLXSGeneticAlgorithm,
    DifferentialEvolutionCTBAlgorithm,
    DifferentialEvolutionRandomAlgorithm,
    IteratedLocalSearchAlgorithm,
    LocalSearchAlgorithm,
    MemeticAlgorithmA,
    MemeticAlgorithmB,
    MemeticAlgorithmC,
    ReliefAlgorithm,
    SimulatedAnnealingAlgorithm
)
from core import Classifier1NN
from utils import ArffReader, Result, ResultsCollector


def main():
    np.seterr(all='raise')  # always raise error, no runtime warnings

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
        BLXEGeneticAlgorithm,
        MemeticAlgorithmC,
        SimulatedAnnealingAlgorithm,
        IteratedLocalSearchAlgorithm,
        DifferentialEvolutionRandomAlgorithm,
        DifferentialEvolutionCTBAlgorithm
    ]

    results = {}

    for alg in algorithms:
        collector = ResultsCollector()

        if args.interactive:
            print('{emph} {alg} {emph}'.format(emph='='*20, alg=alg.__name__))

        for db in databases:
            np.random.seed(0)  # reset NumPy PRNG seed

            dataset = ArffReader.read(db)
            partitions = dataset.kfoldcv(K=5)

            if args.interactive:
                print('{emph} {db} {emph}'.format(emph='*'*10, db=db))

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

                # calculate errors and scores

                res.train_score = learner.classifier.evaluate_solution(
                    learner.solution
                )
                res.train_error = learner.solution.error
                res.test_error = learner.classifier.test_error(
                    test_dataset=partition.testing_set,
                    w=learner.solution.w
                )
                res.test_score = np.mean([1 - res.test_error, learner.solution.redux])

                # set up Result

                res.solution = learner.solution.w.tolist()  # cast np array to list


                if args.interactive:
                    print(res)

                collector.append_result(res)

        results[alg.__name__] = collector

    # transform to something JSON can work with

    results = {
        alg: collector.serialize()
        for alg, collector in results.iteritems()
    }

    if args.output_filename is None:
        print json.dumps(results)
    else:
        f = open(args.output_filename, 'wb')
        f.write(json.dumps(results))
        f.close()


if __name__ == '__main__':
    # prof = cProfile.Profile()

    # prof.enable()
    # main()
    # prof.disable()

    # prof.print_stats(sort='time')

    main()
