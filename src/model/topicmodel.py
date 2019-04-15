from itertools import product
from time import time

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import LatentDirichletAllocation

from util import proj as dt



from common.SaveLoadPOPO import SaveLoadPOPO
from common.Method import Method
from sklearn import preprocessing
class TopicModel(Method):
    popo_example = None
    output_folder = None
    multidim_array = None
    normalized_output = None
    model_words = None
    model_rep = None
    model = None

    def __init__(self, bow,words, doc_topic_prior,  topic_word_prior, dim, file_name, output_folder, save_class):

        self.doc_topic_prior = doc_topic_prior
        self.topic_word_prior = topic_word_prior
        self.file_name = file_name
        self.dim = dim
        self.output_folder = output_folder
        self.save_class = save_class
        self.bow =bow
        self.words=words

        super().__init__(file_name, save_class)

    def makePopos(self):
        self.model_words = SaveLoadPOPO(self.model_words, self.output_folder + self.file_name + ".npy", "npy")
        self.model_rep = SaveLoadPOPO(self.model_rep, self.output_folder + self.file_name + ".npy", "npy")
        self.model =  SaveLoadPOPO(self.model, self.output_folder + self.file_name + ".bin", "joblib")

    def makePopoArray(self):
        self.popo_array = [self.model_words, self.model_rep, self.model]

    def process(self):
        # Process
        lda = LatentDirichletAllocation(doc_topic_prior=self.doc_topic_prior, topic_word_prior=self.topic_word_prior,
                                        n_topics=self.dim)
        t0 = time()
        print(self.bow.shape[0], self.bow.shape[1])
        new_rep = lda.fit_transform(self.bow)
        print("done in %0.3fs." % (time() - t0))

        print("\nTopics in LDA model:")
        self.model_words = self.print_top_words(lda, self.words)
        self.model_words.reverse()
        self.model_rep = new_rep.transpose()
        self.model = lda
        print("completed")
        super().process()

    def getRep(self):
        if self.processed is False:
            self.model_rep.value = self.save_class.load(self.model_rep)
        return self.model_rep.value


    def print_top_words(self, model, feature_names):
        names = []
        for topic_idx, topic in enumerate(model.components_):
            message = ""
            message += " ".join([feature_names[i]
                                 for i in topic.argsort()])
            print(message[:100])
            names.append(message)
        print()
        return names
