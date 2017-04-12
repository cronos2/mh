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
    prof = cProfile.Profile()

    prof.enable()
    main()
    prof.disable()

    prof.print_stats(sort='time')
