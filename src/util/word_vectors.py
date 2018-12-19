
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
def getWordVectors(vector_save_fn, words_fn, wvn, wv_amt, svm_dir_fn=None):
    if os.path.exists(vector_save_fn) is False:
        glove_file = datapath('/home/tom/Downloads/glove.6B/glove.6B.'+str(wv_amt)+'d.txt')
        tmp_file = get_tmpfile("/home/tom/Downloads/glove.6B/test_word2vec.txt")
        glove2word2vec(glove_file, tmp_file)
        svm_dir = import2dArray(svm_dir_fn)
        all_vectors = KeyedVectors.load_word2vec_format(tmp_file)
        vectors = []

        words = import1dArray(words_fn)
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

        write2dArray(vectors, vector_save_fn)


        write1dArray(words, wvn)
    else:
        print("Already got word vectors", vector_save_fn)



def averageWordVectors(id2word, ppmi_fn, size, data_type):
    bow = import2dArray(ppmi_fn)

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


def averageWordVectorsFreq(id2word, freq_fn, size, data_type):
    glove_file = datapath("D:\Downloads\Work/glove.6B/glove.6B." + str(size) + 'd.txt')
    tmp_file = get_tmpfile("D:\Downloads\Work/glove.6B/test_word2vec.txt")
    bow = import2dArray(freq_fn, "i")

    print("Transposing PPMI")
    bow = bow.transpose()
    if len(bow[0]) != len(id2word.keys()):
        print("vocab and bow dont match", len(bow[0]), len(id2word.keys()))
        exit()
    print("Creating dict")
    print("Importing word vectors")
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
                    to_average.append(all_vectors.get_vector(id2word[w]))
                except KeyError:
                    print("keyerror", id2word[w])
        if len(to_average) == 0:
            to_average = [np.zeros(shape=size)]
            print("FAILED", i, "words:", len(to_average), "dim", len(to_average[0]))
        else:
            print(i, "words:", len(to_average), "dim", len(to_average[0]))
        vectors.append(np.average(to_average, axis=0))
        i += 1

    np.save("../data/" + data_type + "/nnet/spaces/wvFIXED" + str(size) + ".npy", vectors)
