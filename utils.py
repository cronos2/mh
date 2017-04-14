# coding: utf-8

import arff
import itertools
import json
import numpy as np
import time

from core import Dataset


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


class Result(object):
    def __init__(self, name=''):
        self.name = name

    def start_timer(self):
        self.clock = time.clock()

    def end_timer(self):
        self.exec_time = time.clock() - self.clock

    def summary(self):
        return {
            'indices': self.indices,  # this should be a dict with Python lists
            'name': self.name,
            'solution': self.solution,  # this should be a plain Python list
            'test_error': self.test_error,
            'time': self.exec_time,
            'train_error': self.train_error,
        }

    def __str__(self):
        return json.dumps(self.summary())


class ResultsCollector(object):
    def __init__(self):
        self.results = []

    def append_result(self, result):
        self.results.append(result)

    def __str__(self):
        return json.dumps([res.summary() for res in self.results])

    def __repr__(self):
        return str(self)


class ResultsReporter(object):
    row_template = ','.join(
            ['{row_name}'],     # title
            ['{:.3f}'] * 3 * 3  # success rate (train+test) and time for each db
        )

    def format_partition(self, i, j, *args, **kwargs):
        row_name = 'Particón {i} - {j}'.format(i=i, j=j)

        return self.row_template.format(row_name=row_name, *args, **kwargs)

    def format_means(self, *args, **kwargs):
        return self.row_template.format(row_name='Media', *args, **kwargs)

    def format_algorithm(self, algorithm, *args, **kwargs):
        return self.row_template.format(row_name=algorithm, *args, **kwargs)

    def read(self, filename):
        f = open(filename, 'rb')
        self.results = json.load(f)
        f.close()

    def report(self):
        output = ''

        for algorithm in self.results:
            current_results = self.results[algorithm]

            sonar = current_results[:10]
            spambase = current_results[10:20]
            wdbc = current_results[20:]
            databases = [sonar, spambase, wdbc]

            for k in xrange(10):
                quotient, remainder = divmod(k, 2)
                row_results = [db[k] for db in databases]

                data = itertools.chain.from_iterable((
                    (1 - res['train_error'],
                     1 - res['test_error'],
                     res['time']) for res in row_results
                ))

                output += self.format_partition(
                    i=quotient+1,
                    j=remainder+1,
                    *data
                ) + '\n'

            reduced_data = np.array([(
                1 - res['train_error'],
                1 - res['test_error'],
                res['time']
            ) for res in current_results]).reshape(3, 10, 3)  # db, part, metric

            means = np.mean(reduced_data, axis=1)  # by columns
            output += self.format_means(means.flat)

        self.report = output
        return output

    def export(self, filename):
        f = open(filename, 'wb')
        f.write(self.report)
        f.close()

    def global_report(self):
        reduced_data = np.array([(
                1 - res['train_error'],
                1 - res['test_error'],
                res['time'])
            for alg in self.results
            for res in self.results[alg]])
        reduced_data = reduced_data.reshape(-1, 3, 10, 3)  # alg, db, part, metric

        means = np.mean(reduced_data, axis=2)  # by columns
        algorithms = self.results.keys()  # the dict hasn't been modified

        output = ''

        for algorithm, values in itertools.izip(algorithms, means):
            output += self.format_algorithm(algorithm, values.flat)

        return output
