import arff
import json
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
    def __init__(self, name = ''):
        self.name = name

    def start_timer(self):
        self.clock = time.clock()

    def end_timer(self):
        self.exec_time = time.clock() - self.clock

    def summary(self):
        return {
            'name': self.name,
            'time': self.exec_time,
            'solution': (self.solution.w.tolist(), self.solution.error)
        }

class ResultsCollector(object):
    def __init__(self):
        self.results = []

    def append_result(self, result):
        self.results.append(result)

    def __str__(self):
        return json.dumps([res.summary() for res in self.results], indent=4)

    def __repr__(self):
        return str(self)
