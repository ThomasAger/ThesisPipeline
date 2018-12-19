from util import io as dt
import numpy as np
import scipy.sparse as sp
from gensim.Corpora import Dictionary

class SaveLoad:

    files = None
    fns = None


    #example_final = {"param_name": (filename, file, file_type)}

    def __init__(self):
        print("init")

    def setFiles(self, fns):
        self.fns = fns

    def setFilesFns(self, files_fns):
        self.files_fns = files_fns

    def load(self,  popo):
        for i in range(len(params)):
            param_dict[params[i]][1] = self.load_by_type(param_dict[params[i]][2], param_dict[params[i]][0])


    def load_by_type(self, type, filename):
        file = None
        if type == "npy":
            file = np.load(filename)
        elif type == "scipy":
            file = sp.load_npz(filename)
        elif type == "gensim":
            file = Dictionary.load(filename)
        elif type == "1dtxts":
            file = dt.import1dArray(filename, "s")
        return file

    def save(self):
        self.dct.save(self.dct_fn)
        np.save(self.remove_ind_fn, self.remove_ind)
        np.save(self.tokenized_corpus_fn, self.tokenized_corpus)
        np.save(self.tokenized_ids_fn, self.tokenized_ids)
        np.save(self.id2token_fn, self.id2token)
        np.save(self.bow_vocab_fn, self.bow_vocab)
        np.save(self.filtered_vocab_fn, self.filtered_vocab)
        dt.write1dArray(self.processed_corpus, self.processed_corpus_fn ,
                        encoding="utf8")
        np.save(self.classes_fn, self.classes)
        if self.classes_categorical is not None: # Idk if this works
            np.save(self.classes_categorical_fn, self.classes_categorical)
        sp.save_npz(self.bow_fn, self.bow)
        sp.save_npz(self.filtered_bow_fn, self.filtered_bow)
        dt.write1dArray(self.word_list, self.word_list_fn, encoding="utf8")
        dt.write1dArray(self.all_words_backup, self.all_words_backup_fn ,
                        encoding="utf8")

    def exists(self):
        return dt.fnsExist(self.fns)