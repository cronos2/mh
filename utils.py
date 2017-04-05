import arff

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
