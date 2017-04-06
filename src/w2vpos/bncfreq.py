from functools import lru_cache

import pandas as pd


class WordPosFreq():
    def __init__(self,data):
        self.data=data

    @lru_cache(maxsize=None)
    def get(self,word):
        posfrq = self.data[self.data.Word == word].sort(['Freq'], ascending=[0])
        if posfrq.shape[0] > 0:
            return posfrq.iloc[0]
        return None

    def get_pos(self):
        return self.data.PoS.unique()

    @staticmethod
    def load(file):
        bncfreq = pd.DataFrame.from_csv(file, header=0, sep='\t', index_col=None)
        col_versions = bncfreq.columns[3]
        bncfreq = bncfreq.drop(bncfreq.columns[0], axis=1)
        bncfreq.loc[bncfreq.Word == '@', 'Word'] = bncfreq[bncfreq.Word == '@'][col_versions]
        roots = bncfreq[bncfreq[col_versions] == '%'].index.tolist()
        while True:
            lacking = bncfreq[bncfreq.PoS == '@'].index.tolist()
            lacking_not = [x - 1 for x in lacking]
            bncfreq.loc[lacking, 'PoS'] = bncfreq.loc[lacking_not, 'PoS'].tolist()
            if len(lacking) == 0:
                break
        bncfreq = bncfreq.drop(bncfreq.index[roots])
        bncfreq = bncfreq.drop(col_versions, axis=1)
        bncfreq.set_index('Word')
        return WordPosFreq(bncfreq)
