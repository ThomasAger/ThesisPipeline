from common import Method
from common.SaveLoadPOPO import SaveLoadPOPO
from util.save_load import SaveLoad
import numpy as np
from model import svm

import scipy.sparse as sp
class GetDirections(Method.Method):
    word_dir = None
    words = None
    words_to_get = None
    output_directions = None
    output_folder = None
    directions = None
    bow = None
    bowmin = None
    bowmax = None
    new_word_dict = None
    space = None
    LR = None
    rewrite_words = None
    dirs_to_label = None
    dirs_to_label_with = None
    dirs_to_label_names = None
    dirs_to_label_with_names = None
    cluster_names = None


    def __init__(self, dirs_to_label, dirs_to_label_with, dirs_to_label_names, dirs_to_label_with_names, save_class,  file_name, output_folder):
        self.words_to_get = words_to_get
        self.output_folder = output_folder
        self.dirs_to_label = dirs_to_label
        self.dirs_to_label_with = dirs_to_label_with
        self.dirs_to_label_names = dirs_to_label_names
        self.dirs_to_label_with_names = dirs_to_label_with_names
        super().__init__(file_name, save_class)

    def makePopos(self):

        self.cluster_names = SaveLoadPOPO({},self.output_folder + self.file_name +"_label_names.npy", "npy_dict")

    def makePopoArray(self):
        if self.rewrite_words is False:
            self.popo_array = [self.word_dir, self.directions, self.predictions, self.new_bow, self.words, self.pred_dir]
        else:
            self.popo_array = [self.words]

    def process(self):
        i = 0
        len_of_list = len(list(self.words_to_get.keys()))
        """
        np_pred = np.empty(shape=(len(list(self.words_to_get.keys()))), dtype=object)
        np_new_bow = np.empty(shape=(len(list(self.words_to_get.keys()))), dtype=object)
        """
        if self.rewrite_words is False:
            for key, value in self.words_to_get.items():
                if key in self.word_dir.value and key in self.pred_dir.value:
                    self.directions.value[self.new_word_dict[key]] = self.word_dir.value[key]
                    self.predictions.value[self.new_word_dict[key]] = self.pred_dir.value[key]
                    self.new_bow.value[self.new_word_dict[key]] = self.bow[value]
                    loaded = True
                else:
                    loaded = False
                    # Get the bow line for the word
                    freq_word_freq = self.bow[value]

                    if np.amax(self.directions.value[self.new_word_dict[key]]) == 0.0 or len(self.predictions.value[self.new_word_dict[key]].data) == 1:
                        word_freq = np.asarray(freq_word_freq.todense(), dtype=np.int32)[0]
                        word_freq[word_freq >= 1] = 1

                        if self.LR is False:
                            dir_svm = svm.LinearSVM(self.space, word_freq, self.space, word_freq, self.file_name, SaveLoad(rewrite=True, no_save=True, verbose=False))
                        else:
                            dir_svm = svm.LogisticRegression(self.space, word_freq, self.space, word_freq, self.file_name, SaveLoad(rewrite=True, no_save=True, verbose=False))

                        dir_svm.process_and_save()
                        self.directions.value[self.new_word_dict[key]] = dir_svm.getDirection()

                        # Sparse
                        self.predictions.value[self.new_word_dict[key]] = dir_svm.getPred()[0]

                    # Sparse
                    self.new_bow.value[self.new_word_dict[key]] = freq_word_freq

                    self.word_dir.value[key] = self.directions.value[self.new_word_dict[key]]
                    self.pred_dir.value[key] = self.predictions.value[self.new_word_dict[key]]
                i += 1
                print(i, "/", len_of_list, key, "loaded", loaded)

            self.predictions.value = sp.csr_matrix(self.predictions.value)
            self.new_bow.value = sp.csr_matrix(self.new_bow.value)


        words = np.empty(len(list(self.new_word_dict.keys())), dtype=object)
        for key, value in self.new_word_dict.items():
            words[value] = key
        self.words.value = np.asarray(list(self.new_word_dict.keys()))
        super().process()

    def getDirections(self):
        if self.processed is False or self.rewrite_words is True:
            self.directions.value = self.save_class.load(self.directions)
        return self.directions.value

    def getDirectionsFn(self):
        return self.directions.file_name

    def getPreds(self):
        if self.processed is False or self.rewrite_words is True:
            self.predictions.value = self.save_class.load(self.predictions)
        return self.predictions.value

    def getNewBow(self):
        if self.processed is False or self.rewrite_words is True:
            self.new_bow.value = self.save_class.load(self.new_bow)
        return self.new_bow.value

    def getWords(self):
        if self.processed is False or self.rewrite_words is True:
            self.words.value = self.save_class.load(self.words)
        return self.words.value

class GetDirectionsSimple(Method.Method):

    words = None
    output_directions = None
    output_folder = None
    directions = None
    bow = None
    new_word_dict = None
    space = None
    LR = None

    def __init__(self, bow, space, save_class,  file_name, output_folder, LR=False):

        self.output_folder = output_folder
        self.space = space
        self.bow = bow
        self.LR = LR
        super().__init__(file_name, save_class)

    def makePopos(self):

        self.directions = SaveLoadPOPO(self.directions, self.output_folder
                                     + "dir/" + self.file_name  + "_dir.npy", "npy")

    def makePopoArray(self):
        self.popo_array = [ self.directions]

    def process(self):
        self.directions.value = []
        for i in range(len(self.bow)):
            self.bow[i][self.bow[i] >= 1] = 1

            if self.LR is False:
                dir_svm = svm.LinearSVM(self.space, self.bow[i], self.space, self.bow[i], self.file_name, SaveLoad(rewrite=True, no_save=True, verbose=False))
            else:
                dir_svm = svm.LogisticRegression(self.space, self.bow[i], self.space, self.bow[i], self.file_name, SaveLoad(rewrite=True, no_save=True, verbose=False))
            dir_svm.process_and_save()
            self.directions.value.append(dir_svm.getDirection())
            print(i, "/", len(self.bow))

        super().process()

    def getDirections(self):
        if self.processed is False:
            self.directions.value = self.save_class.load(self.directions)
        return self.directions.value


if __name__ == '__main__':
    words_to_get = []
    save_class = SaveLoad(rewrite=True)
    get_dir = GetDirections(words_to_get, save_class, "test", "../../data/processed/directions/")
    get_dir.process_and_save()
    directions = get_dir.getDirections()
