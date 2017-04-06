import logging
import nltk
from os.path import *
from gensim.models import Word2Vec
from nltk.corpus import brown, movie_reviews, treebank

from w2vpos.encoder import W2VEncoder
from w2vpos.glove import W2V
from w2vpos.dataset import sts
from w2vpos.tools import *
from w2vpos.optimizer import POSOptimizer

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger('w2v-pos')

nltk.download('punkt',
              'stopwords',
              'averaged_perceptron_tagger',
              'universal_tagset',
              'wordnet')

tagset = ["ADJ", "ADP", "ADV", "CONJ", "DET", "NOUN", "NUM", "PRT", "PRON", "VERB", ".", "X"]

optimizer_settings = {"verbose":True,"n": 50, "ngen": 1000}

datasets = {
    "STS-2012-test": sts('resources/STS/2012-en-test').load(),
    "STS-2013-test": sts('resources/STS/2013-en-test').load(),
    "STS-2014-test": sts('resources/STS/2014-en-test').load(),
    #"STS-2015-test": sts('resources/STS/2015-en-test').load(),
    #"STS-2016-test": sts('resources/STS/2016-en-test').load()
}
vectors = VectorCaching('temp')

configuration = {
    "W2V-GLOVE-300": {
        "encoder": vectors.glove(300),
        "optimizer": optimizer_settings
    },
    "W2V-TREEBANK": {
        "encoder": vectors.treebank(),
        "optimizer": optimizer_settings
    },
    "W2V-MOVIE-REVIEWS": {
        "encoder": vectors.movie_reviews(),
        "optimizer": optimizer_settings
    },
    "W2V-BROWN": {
        "encoder": vectors.brown(),
        "optimizer": optimizer_settings
    }
}
for datasetname, data in datasets.items():
    for name, settings in configuration.items():
        workdir = join('log',datasetname,name)
        caching = Caching(workdir)
        str1_tokens, str2_tokens = caching.tokenize(data)
        str1_encoded, str2_encoded = caching.encode(settings["encoder"], str1_tokens, str2_tokens)
        optimizer = POSOptimizer(
            workdir,
            **settings['optimizer'],
            **{'tagset': tagset,
               'data': {
                   'n': min(10000, len(data['GS'])),
                   'gs': data['GS'],
                   'str1': {'tokens': str1_tokens, 'encoding': str1_encoded},
                   'str2': {'tokens': str2_tokens, 'encoding': str2_encoded}
               }}
        )
        print('No weights: %f' % optimizer.eval([1.0 for p in optimizer.tagset]))
        results = optimizer.fit()
        best = results.max()
        print('Max: %s' % best)
        # Base: No weights: 0.497885
        weights_dict = dict(zip(optimizer.tagset, best.individual))
