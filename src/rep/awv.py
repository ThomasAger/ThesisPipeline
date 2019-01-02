
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np
import os
from util import io as dt
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



def averageWordVectors(id2word, ppmi_fn, size, data_type):
    bow = dt.import2dArray(ppmi_fn)

    if len(bow[0]) != len(id2word.keys()):
        print("vocab and bow dont match", len(bow[0]), len(id2word.keys()))
        exit()
    print("Creating dict")
    print("Importing word vectors")
    glove_file = datapath("D:/Dropbox/PhD/My Work/Code/Paper 2/data/raw/glove/glove.6B." + str(size) + 'd.txt')
    tmp_file = get_tmpfile("D:/Dropbox/PhD/My Work/Code/Paper 2/data/raw/glove/test_word2vec.txt")
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


def averageWordVectorsFreq(word_lists, word_vectors):

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
            to_average = [np.zeros(shape=len(word_vectors[0]))]
            print("No words found", i, "words:", len(to_average), "dim", len(to_average[0]))
        else:
            print(i, "orig words", len(word_lists[i]), "words:", len(to_average), "dim", len(to_average[0]))
        vectors.append(np.average(to_average, axis=0))

    return vectors

def get_word_vectors(size, wv_path=None):
    if wv_path is None:
        glove_file = datapath("D:\Downloads\Work/glove.6B/glove.6B." + str(size) + 'd.txt')
        tmp_file = get_tmpfile("D:\Downloads\Work/glove.6B/test_word2vec.txt")
    else:
        glove_file = datapath(wv_path)
        tmp_file = get_tmpfile(wv_path + "temp.txt")

    glove2word2vec(glove_file, tmp_file)

    all_vectors = KeyedVectors.load_word2vec_format(tmp_file)
    return all_vectors



from common.SaveLoadPOPO import SaveLoadPOPO
from common.Method import Method
class PCA(Method):
    rep = None
    word_doc_matrix = None
    dim = None
    file_name = None

    def __init__(self,  id2word_dict,wv_path, dim, file_name, save_class):

        self.word_doc_matrix = word_doc_matrix
        self.checkWordDocMatrix(doc_amt)
        self.dim = dim
        self.file_name = file_name

        super().__init__(file_name, save_class)

    def checkWordDocMatrix(self, doc_amt):
        if self.word_doc_matrix.shape[1] != doc_amt and self.word_doc_matrix.shape[0] != doc_amt:
            raise ValueError("Incorrect number of documents")
        # Check if the words, typically the more frequent, are the rows or the columns, and transpose so they are the columns
        if self.word_doc_matrix.shape[1] == doc_amt:
            self.word_doc_matrix = self.word_doc_matrix.transpose()

    def makePopos(self):
        self.rep = SaveLoadPOPO(self.rep, self.file_name + "PCA.npy", "npy")

    def makePopoArray(self):
        self.popo_array = [self.rep]

    def process(self):
        print("Begin processing")
        svd = averageWordVectors(n_components=self.dim)  # use the scipy algorithm "arpack"
        self.rep.value = svd.fit_transform(self.word_doc_matrix)
        super().process()
