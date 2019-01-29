from common import Method
from common.SaveLoadPOPO import SaveLoadPOPO
from util.save_load import SaveLoad
import numpy as np
from model import svm


class GetDirections(Method.Method):
    word_dir = None
    words_to_get = None
    output_directions = None
    output_folder = None
    directions = None
    bow = None
    bowmin = None
    bowmax = None
    new_word_dict = None
    corpus = None

    def __init__(self, bow, corpus,  words_to_get, new_word_dict, save_class, bowmin, bowmax, file_name, output_folder):
        self.words_to_get = words_to_get
        self.output_folder = output_folder
        self.corpus = corpus
        self.bow = bow
        self.bowmin = bowmin
        self.new_word_dict = new_word_dict
        self.bowmax = bowmax
        super().__init__(file_name, save_class)

    def makePopos(self):
        self.word_dir = SaveLoadPOPO({},self.output_folder + self.file_name +"_word_dir.npy", "npy_dict")
        if self.save_class.exists([self.word_dir]):
            self.word_dir.value = self.save_class.load(self.word_dir)
            print("word_dir loaded successfully")

        self.directions = SaveLoadPOPO(np.empty(len(list(self.words_to_get.keys())), dtype=object), self.output_folder
                                       + self.file_name + "_" + str(self.bowmin) + "_" + str(self.bowmax) + "_""_dir.npy", "npy")

        self.predictions = SaveLoadPOPO(np.empty(len(list(self.words_to_get.keys())), dtype=object), self.output_folder
                                       + self.file_name + "_" + str(self.bowmin) + "_" + str(self.bowmax) + "_""_pred.npy", "npy")
        self.new_bow = SaveLoadPOPO(np.empty(len(list(self.words_to_get.keys())), dtype=object), self.output_folder
                                       + self.file_name + "_" + str(self.bowmin) + "_" + str(self.bowmax) + "_""_bow.npz", "scipy")

    def makePopoArray(self):
        self.popo_array = [self.word_dir, self.directions, self.new_bow]

    def process(self):
        i = 0
        len_of_list = len(list(self.words_to_get.keys()))
        for key, value in self.words_to_get.items():
            try:
                self.directions.value[self.new_word_dict[key]] = self.word_dir.value[key]
            except KeyError:
                # Get the bow line for the word
                freq_word_freq = self.bow[value]
                word_freq = np.asarray(freq_word_freq.todense(), dtype=np.int32)[0]
                word_freq[word_freq > 1] = 1
                dir_svm = svm.LinearSVM(self.corpus, word_freq, self.corpus, word_freq, self.file_name, SaveLoad(rewrite=True, no_save=True, verbose=False))
                dir_svm.process_and_save()
                self.directions.value[self.new_word_dict[key]] = dir_svm.getDirection()
                self.predictions.value[self.new_word_dict[key]] = dir_svm.getPred()
                self.word_dir.value[key] = self.directions.value[self.new_word_dict[key]]
                self.new_bow.value[self.new_word_dict[key]] = freq_word_freq
            i += 1
            print(i, "/", len_of_list, key)
        super().process()

    def getDirections(self):
        if self.save_class.exists([self.directions]) is True and self.directions.value is None:
            self.directions.value = self.save_class.load(self.directions)
        return self.directions.value

    def getNewBow(self):
        if self.save_class.exists([self.new_bow]) is True and self.new_bow.value is None:
            self.new_bow.value = self.save_class.load(self.new_bow)
        return self.new_bow.value

if __name__ == '__main__':
    words_to_get = []
    save_class = SaveLoad(rewrite=True)
    get_dir = GetDirections(words_to_get, save_class, "test", "../../data/processed/directions/")
    get_dir.process_and_save()
    directions = get_dir.getDirections()
