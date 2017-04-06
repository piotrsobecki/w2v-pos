import os
import re
from nltk.tokenize import word_tokenize
import pandas as pd
from os.path import exists


class sts():
    def __init__(self, dir):
        self.dir = dir

    def load(self):
        pattrn = r'STS.gs.(.+).txt'
        gs = [f for f in os.listdir(self.dir) if re.match(pattrn, f)]
        tasks = []
        for f in gs:
            task = re.search(pattrn, f, re.IGNORECASE).group(1)
            task_df = self.load_task(task)
            task_df.Task = task
            tasks.append(task_df)
        return pd.concat(tasks)

    def load_task(self, task):
        format = 'STS.%s.%s.txt'
        input_f = os.path.join(self.dir, format % ('input', task))
        gs_f = os.path.join(self.dir, format % ('gs', task))
        str1,str2,gs = [],[],[]
        if exists(input_f) and exists(gs_f):
            str1, str2 = self.load_data(input_f)
            with open(gs_f) as f:
                gs = [float(gs) for gs in f.readlines()]
        return pd.DataFrame({'Str1': str1, 'Str2': str2, 'GS': gs})

    def load_data(self, testloc):
        trainA, trainB, testA, testB = [], [], [], []
        f = open(testloc, 'r', encoding='UTF-8')
        for line in f:
            text = line.strip().split('\t')
            testA.append(' '.join(word_tokenize(text[0])))
            testB.append(' '.join(word_tokenize(text[1])))
        f.close()
        return [testA, testB]
