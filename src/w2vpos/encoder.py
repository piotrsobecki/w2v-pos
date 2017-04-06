from functools import lru_cache

import numpy as np
import logging


class W2VEncoder():
    def __init__(self, w2v):
        self.logger = logging.getLogger('w2v-pos')
        self.w2v = w2v

    def encode(self, tokens):
        rows = len(tokens)
        cols = self.w2v.size()
        sense = np.zeros([rows, cols])
        for i in range(len(tokens)):
            try:
                sense[i] = self.w2v.get(tokens[i][0])
            except KeyError:
                self.logger.warning('No word: %s' % tokens[i][0])
        return sense


class Weights():

    default_weights = {None: 0.5}

    def __init__(self, weights=None):
        if weights is None:
            weights = {}
        self.weights = {**self.default_weights,**weights}

    @lru_cache(maxsize=None)
    def weight(self, pos=None):
        pos_out = None
        if pos in self.weights:
            pos_out = pos
        return self.weights[pos_out]
