from common.SaveLoadPOPO import SaveLoadPOPO
from common.Method import Method
from sklearn import preprocessing
from scipy import sparse as sp
import numpy as np

class Filter(Method):
    bow = None
    words_to_get = None
    output_folder = None
    new_word_dict = None
    new_bow = None
    words = None

    def __init__(self, bow, words_to_get, new_word_dict, file_name, save_class, output_folder):

        self.bow = bow
        self.words_to_get = words_to_get
        self.new_word_dict = new_word_dict
        self.output_folder = output_folder

        super().__init__(file_name, save_class)

    def makePopos(self):
        self.new_bow = SaveLoadPOPO(self.new_bow, self.output_folder + self.file_name + ".npy", "npy")
        self.words = SaveLoadPOPO(self.words, self.output_folder + self.file_name + ".npy", "npy")

    def makePopoArray(self):
        self.popo_array = [self.new_bow]

    def process(self):
        # Process
        shape = (len(self.words_to_get.keys()), self.bow.shape[1])
        empty_data = np.empty([shape[0], shape[1]])
        self.new_bow.value = sp.lil_matrix(empty_data, shape=shape)

        for key, value in self.words_to_get.items():
            self.new_bow.value[self.new_word_dict[key]] = self.bow[value]

        self.words.value = np.empty(len(list(self.new_word_dict.keys())), dtype=np.object_)

        for key, value in self.new_word_dict.items():
            self.words.value[value] = key

        self.new_bow.value = np.asarray(self.new_bow.value.todense())
        print("completed")
        super().process()

    def getNewBow(self):
        if self.processed is False:
            self.new_bow.value = self.save_class.load(self.new_bow)
        return self.new_bow.value

    def getWords(self):
        if self.processed is False:
            self.words.value = self.save_class.load(self.words)
        return self.words.value