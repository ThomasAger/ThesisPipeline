
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np
import os
from util import io as dt
""" OLD 
def getWordVectors(vector_save_fn, words_fn, wvn, wv_amt, svm_dir_fn=None):
    if os.path.exists(vector_save_fn) is False:
        glove_file = datapath('/home/tom/Downloads/glove.6B/glove.6B.'+str(wv_amt)+'d.txt')
        tmp_file = get_tmpfile("/home/tom/Downloads/glove.6B/test_word2vec.txt")
        glove2word2vec(glove_file, tmp_file)
        svm_dir = dt.import2dArray(svm_dir_fn)
        all_vectors = KeyedVectors.load_word2vec_format(tmp_file)
        vectors = []

        words = dt.import1dArray(words_fn)
        for w in range(len(words)):
            try:
                if svm_dir_fn is None:
                    vectors.append(all_vectors.get_vector(words[w]))
                else:
                    vectors.append(np.concatenate([all_vectors.get_vector(words[w]), svm_dir[w]]))
            except KeyError:
                if svm_dir_fn is None:
                    vectors.append(np.zeros(wv_amt))
                else:
                    vectors.append(np.zeros(wv_amt + len(svm_dir[0])))

        dt.write2dArray(vectors, vector_save_fn)


        dt.write1dArray(words, wvn)
    else:
        print("Already got word vectors", vector_save_fn)
"""

""" OLD
def averageWordVectors(id2word, ppmi_fn, size, data_type):
    bow = dt.import2dArray(ppmi_fn)

    if len(bow[0]) != len(id2word.keys()):
        print("vocab and bow dont match", len(bow[0]), len(id2word.keys()))
        exit()
    print("Creating dict")
    print("Importing word vectors")
    glove_file = datapath("../../data/raw/glove/glove.6B." + str(size) + 'd.txt')
    tmp_file = get_tmpfile("../../data/raw/glove/glove.6B." + str(size) + "dtemp.txt")
    glove2word2vec(glove_file, tmp_file)

    all_vectors = KeyedVectors.load_word2vec_format(tmp_file)
    print("Creating vectors")
    vectors = []
    i = 0
    for doc in bow:
        to_average = []
        for w in range(len(doc)):
            if doc[w] > 0:
                try:
                    to_average.append(np.multiply(all_vectors.get_vector(id2word[w]), doc[w]))
                except KeyError:
                    print("keyerror", id2word[w])
        if len(to_average) == 0:
            to_average = [np.zeros(shape=size)]
            print("FAILED", i, "words:", len(to_average), "dim", len(to_average[0]))
        else:
            print(i, "words:", len(to_average), "dim", len(to_average[0]))
        vectors.append(np.average(to_average, axis=0))
        i+=1

    np.save("../data/" +data_type+"/nnet/spaces/wvPPMIFIXED" + str(size) + ".npy", vectors)
 """

def getAWV(word_lists, word_vectors):

    print("Creating vectors")
    vectors = []
    for i in range(len(word_lists)):
        to_average = []
        for j in range(len(word_lists[i])):
            try:
                to_average.append(word_vectors.get_vector(word_lists[i][j]))
            except KeyError:
                print("keyerror", word_lists[i][j])
        if len(to_average) == 0:
            # Just chose a random word-vector here that I was sure will exist
            #to_average = [np.zeros(shape=len(word_vectors["dog"]))]
            print("No words found", i, "words:", len(to_average), "dim", len(to_average[0]))
        else:
            print(i, "orig words", len(word_lists[i]), "words:", len(to_average), "dim", len(to_average[0]))
            vectors.append(np.average(to_average, axis=0))

    return np.asarray(vectors)

import scipy.sparse as sp
import math
def getAWVw(ppmi, id2word, word_vectors):
    ppmi = ppmi.toarray()
    vectors = []
    for i in range(len(ppmi)):
        amt = 0
        to_divide = np.zeros(shape=len(word_vectors["dog"]))
        for j in range(len(ppmi[i])):
            if ppmi[i][j] > 0:
                amt += 1
                try:
                    to_divide = np.add(to_divide, np.multiply(word_vectors.get_vector(id2word[j]), ppmi[i][j]))
                except KeyError:
                    print("keyerror", id2word[j])
        if len(to_divide) == 0:
            to_divide = np.zeros(shape=len(word_vectors["dog"]))
            print("FAILED", i, "words:", len(to_divide), "dim", len(to_divide[0]))
        else:
            print(i, "words:", len(to_divide))
        if np.sum(to_divide) == 0:
            divided = np.zeros(shape=len(word_vectors["dog"]))
        else:
            divided = np.divide(to_divide, amt) #np.sum(ppmi[i])
        vectors.append(divided)
        i+=1

    return np.asarray(vectors)

""" OLD
def getAWVwSparse(ppmi, id2word, word_vectors):
    print("Creating vectors")
    vectors = []
    ppmi = sp.coo_matrix(ppmi)
    current_i = -1337
    sum = 0
    for i, j, v in zip(ppmi.row, ppmi.col, ppmi.data):
        print(i, j, v)
        if i != current_i:
            if current_i != -1337:
                vectors.append(np.divide(to_divide, sum))
            to_divide = np.zeros(shape=len(word_vectors["dog"]))
            sum = 0
        try:
            print(to_divide, id2word[j], v)
            to_divide = np.add(to_divide, np.multiply(word_vectors.get_vector(id2word[j]), v))
            sum += v
        except KeyError:
            print("keyerror", id2word[j])
        current_i = i
    return np.asarray(vectors)
"""

def get_word_vectors(size, wv_path=None):
    if wv_path is None:
        glove_file = datapath("D:\\PhD\\Code\\ThesisPipeline\\ThesisPipeline\\data\\raw\glove\\glove.6B." + str(size) + 'd.txt')
        tmp_file = get_tmpfile("D:\\PhD\\Code\\ThesisPipeline\\ThesisPipeline\\data\\raw\glove\\glove.6B." + str(size) + 'dtemp.txt')
    else:
        glove_file = datapath(wv_path)
        tmp_file = get_tmpfile(wv_path + "temp.txt")

    glove2word2vec(glove_file, tmp_file)

    all_vectors = KeyedVectors.load_word2vec_format(tmp_file)
    return all_vectors



from common.SaveLoadPOPO import SaveLoadPOPO
from common.Method import RepMethod
class AWV(RepMethod):

    word_lists = None
    wv_path = None

    def __init__(self, word_lists, dim, file_name, output_folder, save_class,  wv_path=None):
        self.word_lists = word_lists
        self.wv_path = wv_path
        super().__init__(file_name, output_folder, save_class, dim)

    def process(self):
        word_vectors = get_word_vectors(self.dim, self.wv_path)
        awv = getAWV(self.word_lists, word_vectors)  # use the scipy algorithm "arpack"
        self.rep.value = awv
        super().process()


class AWVw(RepMethod):

    ppmi = None
    id2word = None
    wv_path = None

    def __init__(self, ppmi, id2word, dim, file_name, output_folder, save_class,  wv_path=None):
        self.id2word = id2word
        self.wv_path = wv_path
        self.ppmi = ppmi
        super().__init__(file_name, output_folder, save_class, dim)

    def process(self):
        word_vectors = get_word_vectors(self.dim, self.wv_path)
        awv = getAWVw(self.ppmi, self.id2word, word_vectors)  # use the scipy algorithm "arpack"
        self.rep.value = awv
        super().process()
