import logging
import os
import numpy as np
from scipy import spatial

from opt.weights.genetic import WeightOptimizer, WeightsLogHelper
from scipy.stats import spearmanr
from w2vpos.encoder import Weights
from deap import  tools


class POSLogHelper(WeightsLogHelper):

    def __init__(self,base_dir,pos):
        super().__init__()
        self.pos = pos
        self.logger = logging.getLogger('w2v-pos')
        logfile = os.path.join(base_dir,'log.csv')
        if os.path.exists(logfile):
            self.log_file = open(logfile, 'a')
        else:
            self.log_file = open(logfile, 'a')
            print(";".join(["Fitness",*self.pos]), file=self.log_file)
            self.log_file.flush()

    def log(self, context, generation_no, results):
        config = results.max()
        self.logger.info('Generation %d (%f): %s' % (generation_no, config.value(), dict(zip(self.pos,config.individual))))
        self.log_individual(config.value(),config.individual)

    def log_individual(self,fitness,individual):
        print(";".join(format(x, "f") for x in [fitness,*individual]), file=self.log_file)
        self.log_file.flush()

    def close(self, context):
        self.log_file.flush()
        self.log_file.close()


class POSOptimizer(WeightOptimizer):
    def __init__(self, base_dir, **settings):
        self.base_dir = base_dir
        self.logger = logging.getLogger('w2v-pos')
        self.data = settings['data']
        self.tagset = settings['tagset']
        settings['n_weights'] = len(self.tagset)
        super().__init__(**settings)

    def on_fit_start(self, context):
        individual = [1.0 for p in self.tagset]
        fitness = self.eval(individual)
        context["log"].log_individual(fitness[0],individual)

    def corr(self, yhat, y):
        return spearmanr(yhat, y)[0]

    def mate(self, toolbox):
        toolbox.register("mate", tools.cxSimulatedBinaryBounded,eta=10,low=0,up=1)

    def mutate(self, toolbox):
        toolbox.register("mutate", tools.mutPolynomialBounded,eta=10,low=0,up=1, indpb=self.indpb)

    def log_helper(self):
        return POSLogHelper(self.base_dir,self.tagset)

    def eval(self, individual):
        try:
            pos_weights = Weights(dict(zip(self.tagset,individual)))
            evals = np.zeros(self.data['n'])
            for i in range(self.data['n']):
                enc1, enc2 = self.data['str1']['encoding'][i], self.data['str2']['encoding'][i]
                tokens1, tokens2 = self.data['str1']['tokens'][i], self.data['str2']['tokens'][i]
                weights1, weights2 = [pos_weights.weight(pos) for word,pos in tokens1], [pos_weights.weight(pos) for word,pos in tokens2]
                try:
                    sent1_mean = np.average(enc1, weights=weights1, axis=0)
                except ZeroDivisionError:
                    sent1_mean = np.average(enc1, axis=0)
                try:
                    sent2_mean = np.average(enc2, weights=weights2, axis=0)
                except ZeroDivisionError:
                    sent2_mean = np.average(enc2, axis=0)
                evals[i] = 1.0 - spatial.distance.cosine(sent1_mean, sent2_mean)
            return self.corr(evals, self.data['gs'][:self.data['n']].values.tolist()),
        except:
            self.logger.exception('Evaluation exception')
            return -float("inf"),
