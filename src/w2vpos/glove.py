import os
import gensim
import smart_open
import tempfile

from os.path import exists


class W2V():
    def __init__(self, vectors):
        self.vectors = vectors

    def get(self, word):
        return self.vectors.word_vec(word)

    def size(self):
        return self.vectors.syn0.shape[1]

    @classmethod
    def load(cls, glove_vector_file, limit=None):
        return cls(cls.load(glove_vector_file,limit))

    @classmethod
    def load_kv(cls, glove_vector_file, limit=None):
        def get_info(glove_file_name):
            with smart_open.smart_open(glove_file_name) as f:
                num_lines = sum(1 for line in f)
            with smart_open.smart_open(glove_file_name) as f:
                num_dims = len(f.readline().split()) - 1
            return num_lines, num_dims

        def prepend_line(infile, outfile, line):
            with open(infile, 'r', encoding="utf8") as old:
                with open(outfile, 'w', encoding="utf8") as new:
                    new.write(str(line.strip()) + "\n")
                    for line in old:
                        new.write(line)
            return outfile

        output_model = os.path.join(tempfile.gettempdir(), '.%s.gensim' % os.path.basename(glove_vector_file))
        if not exists(output_model):
            num_lines, dims = get_info(glove_vector_file)
            gensim_first_line = "{} {}".format(num_lines, dims)
            prepend_line(glove_vector_file, output_model, gensim_first_line)
        return gensim.models.KeyedVectors.load_word2vec_format(output_model, limit=limit)  # GloVe Model
