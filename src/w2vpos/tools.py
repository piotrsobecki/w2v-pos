import json
import os
import logging
import pickle

from os.path import join, dirname, exists

import nltk
import numpy
from gensim.models import Word2Vec

from w2vpos.encoder import W2VEncoder, Weights
from w2vpos.glove import W2V
from w2vpos.bncfreq import WordPosFreq
from w2vpos.dataset import sts
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, treebank, movie_reviews, brown

logger = logging.getLogger('w2v-pos')

def download_deps():
    nltk.download('punkt',
                  'stopwords',
                  'averaged_perceptron_tagger',
                  'universal_tagset'
                  'movie_reviews')

def tokenize(str1, str2,stopwords):
    try:
        str1pos = nltk.pos_tag(word_tokenize(str1), tagset='universal')
        str2pos = nltk.pos_tag(word_tokenize(str2), tagset='universal')
        return str1pos, str2pos
    except TypeError:
        if '\t' in str1:
            str = str1.split('\t')
            return tokenize(str[0], str[1],stopwords)
        if '\t' in str2:
            str = str2.split('\t')
            return tokenize(str[0], str[1],stopwords)
        logger.exception('Type error')
        return None

def tokenize_data(data):
    str1_tokens = []
    str2_tokens = []
    for idx, row in data.iterrows():
        str1words, str2words = [], []
        try:
            str1words, str2words = tokenize(row.Str1, row.Str2,set(stopwords.words('english')))
        except:
            logger.exception('Tokenizing error')
        str1_tokens.append(str1words)
        str2_tokens.append(str2words)
    return str1_tokens,str2_tokens


def encode_data(encoder,str1_tokens, str2_tokens):
    str1_encoded = []
    str2_encoded = []
    for str1words, str2words in zip(str1_tokens, str2_tokens):
        enc1, enc2 = [], []
        try:
            enc1, enc2 = encoder.encode(str1words), encoder.encode(str2words)
        except:
            logger.exception('Encoding error')
        str1_encoded.append(enc1)
        str2_encoded.append(enc2)
    return str1_encoded,str2_encoded

class VectorCaching():

    def __init__(self,base_dir):
        if not exists(base_dir):
            os.makedirs(base_dir)
        self.base_dir=base_dir

    def treebank(self):
        file = join(self.base_dir,'treebank')
        if exists(file):
            kv = pickle.load(open(file, "rb"))
        else:
            kv = Word2Vec(treebank.sents()).wv
            pickle.dump(kv, open(file, "wb"))
        return W2VEncoder(W2V(kv))

    def movie_reviews(self):
        file = join(self.base_dir, 'movie_reviews')
        if exists(file):
            kv = pickle.load(open(file, "rb"))
        else:
            kv = Word2Vec(movie_reviews.sents()).wv
            pickle.dump(kv, open(file, "wb"))
        return W2VEncoder(W2V(kv))

    def brown(self):
        file = join(self.base_dir, 'brown')
        if exists(file):
            kv = pickle.load(open(file, "rb"))
        else:
            kv = Word2Vec(brown.sents()).wv
            pickle.dump(kv, open(file, "wb"))
        return W2VEncoder(W2V(kv))

    def glove(self,n=300):
        file = join(self.base_dir, 'glove-%d'%n)
        if exists(file):
            kv = pickle.load(open(file, "rb"))
        else:
            kv = W2V.load_kv('resources/word2vec/glove.6B.%dd.txt'%n)
            pickle.dump(kv, open(file, "wb"))
        return W2VEncoder(W2V(kv))


class Caching():

    def __init__(self,base_dir):
        if not exists(base_dir):
            os.makedirs(base_dir)
        self.base_dir=base_dir
        self.tokenized = join(self.base_dir, 'tokenized')
        self.encoded = join(self.base_dir, 'encoded')
        self.weights_file = join(self.base_dir, 'weights')

    def tokenize(self,data):
        if exists(self.tokenized):
            str1_tokens, str2_tokens = pickle.load(open(self.tokenized, "rb"))
        else:
            str1_tokens, str2_tokens = tokenize_data(data)
            pickle.dump((str1_tokens, str2_tokens), open(self.tokenized, "wb"))
        return str1_tokens, str2_tokens

    def encode(self,encoder,str1_tokens,str2_tokens):
        if exists(self.encoded):
            str1_encoded, str2_encoded = pickle.load(open(self.encoded, "rb"))
        else:
            str1_encoded, str2_encoded = encode_data(encoder, str1_tokens, str2_tokens)
            pickle.dump((str1_encoded, str2_encoded), open(self.encoded, "wb"))
        return str1_encoded,str2_encoded