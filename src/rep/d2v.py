
import gensim.models as g
from common.Method import RepMethod
from common.SaveLoadPOPO import SaveLoadPOPO
import logging
import numpy as np
# Defaults set to the Q-Dup model
def doc2Vec(embedding_fn, corpus_fn, vector_size=300, window_size=15, min_count=5, sampling_threshold=1e-5,
                negative_size=5, train_epoch=20, dm=0, worker_count=1, train_wv=1, concatenate_wv=1):

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    docs = g.doc2vec.TaggedLineDocument(corpus_fn) #
    model = g.Doc2Vec(docs, size=vector_size,window=window_size, iter=train_epoch, min_count=min_count,
                        sample=sampling_threshold,negative=negative_size,
                      workers=worker_count,  dm=dm, dbow_words=train_wv, dm_concat=concatenate_wv,
                      pretrained_emb=embedding_fn)
    return model

class D2V(RepMethod):
    embedding_fn = None
    d2v_model = None
    rep = None
    corpus_fn = None
    window_size = 0
    train_epoch = 0
    min_count = 0

    def __init__(self, corpus_fn, embedding_fn, file_name, output_folder, save_class, dim, window_size=15, train_epoch=20, min_count=5):
        self.embedding_fn = embedding_fn
        self.corpus_fn = corpus_fn
        self.window_size = window_size
        self.train_epoch = train_epoch
        self.min_count = min_count
        super().__init__(file_name, output_folder, save_class, dim)

    def makePopoArray(self):
        self.popo_array = [self.d2v_model, self.rep]

    def makePopos(self):
        self.d2v_model = SaveLoadPOPO(self.d2v_model, self.output_folder + self.file_name + ".bin", "gensim_save_model")
        super().makePopos()

    def process(self):
        self.d2v_model.value = doc2Vec(self.embedding_fn, vector_size=self.dim, corpus_fn=self.corpus_fn, worker_count=5, window_size=self.window_size, train_epoch=self.train_epoch, min_count=self.min_count)
        self.rep.value = []
        for d in range(len(self.d2v_model.value.docvecs)):
            self.rep.value.append(self.d2v_model.value.docvecs[d])
        self.rep.value = np.asarray(self.rep.value)

""" OLD
def main(data_type, vector_size, window_size, min_count, sampling_threshold, negative_size,
                               train_epoch, dm, worker_count, train_wv, concatenate_wv, use_hierarchical_softmax):
    file_name = "Doc2Vec" + " VS" + str(vector_size) + " WS" + str(window_size) + " MC" + str(min_count) + " ST" + str(
        sampling_threshold) + \
                " NS" + str(negative_size) + " TE" + str(train_epoch) + " DM" + str(dm) + " WC" + str(
        worker_count) + "spacy"
    " NS" + str(negative_size) + " TE" + str(train_epoch) + " DM" + str(dm) + " WC" + str(worker_count) + \
    " TW" + str(train_wv) + " CW" + str(concatenate_wv) + " HS" + str(use_hierarchical_softmax)

    corpus_fn = "../data/raw/" + data_type + "/corpus_processed.txt"

    if os.path.exists(corpus_fn) is False:
        x_train = np.load("../data/raw/" + data_type + "/x_train_w.npy")
        x_test = np.load("../data/raw/" + data_type + "/x_test_w.npy")
        corpus = np.concatenate((x_train, x_test), axis=0)
        text_corpus = np.empty(len(corpus), dtype=np.object)
        for i in range(len(corpus)):
            text_corpus[i] = " ".join(corpus[i])
            print(text_corpus[i])
        dt.write1dArray(text_corpus, corpus_fn)

    embedding_fn = "/home/tom/Downloads/glove.6B/glove.6B.300d.txt"

    model_fn = "../data/" + data_type + "/doc2vec/" + file_name + ".bin"
    vector_fn = "../data/" + data_type + "/nnet/spaces/" + file_name + ".npy"
    score_fn = "../data/" + data_type + "/doc2vec/" + file_name + "catacc.score"

    if os.path.exists(model_fn):
        print("Imported model")
        model = g.utils.SaveLoad.load(model_fn)
    elif file_name[:7] == "Doc2Vec":
        model = doc2Vec(embedding_fn, corpus_fn, vector_size, window_size, min_count, sampling_threshold,
                        negative_size, train_epoch, dm, worker_count, train_wv, concatenate_wv, use_hierarchical_softmax)
        model.save(model_fn)

    if os.path.exists(vector_fn) is False:
        vectors = []
        for d in range(len(model.docvecs)):
            vectors.append(model.docvecs[d])
        np.save(vector_fn, vectors)
    else:
        print("Imported vectors")
        vectors = np.load(vector_fn)

    if os.path.exists(score_fn) is False or file_name[:6] != "Doc2Vec":
        print("Getting score")
        if data_type == "sentiment":
            classes = dt.import1dArray("../data/" + data_type + "/classify/" + data_type + "/class-all", "i")
            x_train, y_train, x_test, y_test = sentiment.getSplits(vectors, classes)
            scores = linearSVMScore(x_train, y_train, x_test, y_test)
        else:
            classes = dt.import2dArray("../data/" + data_type + "/classify/" + data_type + "/class-all", "i")
            x_train, y_train, x_test, y_test = newsgroups.getSplits(vectors, classes)
            scores = multiClassLinearSVM(x_train, y_train, x_test, y_test)
        print(scores)
        dt.write1dArray(scores, score_fn)

#https://arxiv.org/pdf/1607.05368.pdf
#Found that dbow is better than dmpv
# Vector size 300, window size 15, min count 5, sub sampling 10-5, negative sample 5, epoch 20 are the "best" results for 4.3m training size
vector_size = 300
window_size = 15
min_count = 5
sampling_threshold = 1e-5
negative_size = 5
train_epoch = 20
dm = 0
worker_count = 10
train_wv = 1
concatenate_wv = 1
use_hierarchical_softmax = 1
data_type = "newsgroups"

if  __name__ =='__main__':main(data_type, vector_size, window_size, min_count, sampling_threshold, negative_size,
                               train_epoch, dm, worker_count, train_wv, concatenate_wv, use_hierarchical_softmax)
"""