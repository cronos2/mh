import json

from utils import ResultsReporter


def main():
    reporter = ResultsReporter()
    reporter.read('results/4_full.txt')

    reporter.boxplots('results/boxplot.csv')


if __name__ == '__main__':
    main()
    # boxplot(data, las=2, names=c('RELIEF', 'Local search', 'ACE', 'ACS', 'BLXE', 'BLXS', 'AM (10, 1)', 'AM (10, 0,1)', 'AM (10, 0.1mej)'), par(mar=c(7.5, 4, 2, 2)), at=c(1,3,5,6,7,8,10,11,12), ylab="Tasa de acierto (%)")
