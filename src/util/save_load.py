
from gensim.corpora import Dictionary
import os
import numpy as np
import util.io as dt
import scipy.sparse as sp
import gensim.models as g
class SaveLoad:
    rewrite = None
    no_save = None
    load_all = None
    verbose = None

    def __init__(self, rewrite=False, no_save=False, load_all=False, verbose=True):
        self.rewrite = rewrite
        self.no_save = no_save
        self.load_all = load_all
        self.verbose = verbose

    def loadAll(self, popo_array):
        for i in range(len(popo_array)):
            if popo_array[i].file_name is None or popo_array[i].file_type is None:
                raise ValueError("None type" + popo_array[i].value.file_name)
            popo_array[i].value = self.load(popo_array[i])
        return popo_array

    def load(self, popo):
        return load_by_type(popo.file_type, popo.file_name)

    def save(self, popo_array):
        if self.no_save is False:
            for i in range(len(popo_array)):
                if popo_array[i].value is None or popo_array[i].file_type is None:
                    raise ValueError("Value or type is none ", popo_array[i].value, popo_array[i].file_name)
                elif popo_array[i].file_name is None:
                    raise ValueError("Value is none" + popo_array[i].value)
                save_by_type(popo_array[i].value, popo_array[i].file_type, popo_array[i].file_name)

    def exists(self, popo_array):
        if self.rewrite:
            if self.verbose:
                print(popo_array[0].file_name, "Rewriting...")
            return False
        all_exist = 0
        for i in range(len(popo_array)):
            if os.path.exists(popo_array[i].file_name):
                if self.verbose:
                    print(popo_array[i].file_name, "Already exists")
                all_exist += 1
            else:
                if self.verbose:
                    print(popo_array[i].file_name, "Doesn't exist")
        if all_exist == len(popo_array):
            return True
        return False


def load_by_type(type, file_name):
    try:
        if type == "npy":
            file = np.load(file_name, allow_pickle=True)
        elif type == "scipy" or type == "npz":
            file = sp.load_npz(file_name)
        elif type == "dct":
            file = dt.load_dict(file_name)
        elif type == "npy_dict":
            file = dt.loadNpyDict(file_name)
        elif type == "joblib":
            joblib.load( file_name)
        elif type == "gensim":
            file = Dictionary.load(file_name)
        elif type == "1dtxts":
            file = dt.import1dArray(file_name, "s")
        elif type == "1dtxtf":
            file = dt.import1dArray(file_name, "f")
        elif type == "txtf":
            file = dt.importValue(file_name, "f")
        elif type == "scoredict":
            file = dt.read_csv(file_name)
        elif type == "scoredictarray":
            file = dt.read_csv(file_name)
        elif type == "csv":
            file = dt.csv_pd_to_array(dt.read_csv(file_name))
        elif type == "gensim_save_model":
            file = g.utils.SaveLoad.load(file_name)
        elif type == "dct":
            dt.load_dict(file_name)
        else:
            raise ValueError("File type not recognized")
    except FileNotFoundError:
        print("File was not found.", file_name)
        return None
    return file

from sklearn.externals import joblib
def save_by_type(file, type, file_name):
    if type == "npy":
        np.save(file_name, file)
    elif type == "joblib":
        joblib.dump(file, file_name)
    elif type == "scipy" or type == "npz":
        sp.save_npz(file_name, file)
    elif type == "gensim":
        file.save(file_name)
    elif type == "dct":
        dt.save_dict(file, file_name)
    elif type == "npy_dict":
        np.save(file_name, file)
    elif type[0:5] == "1dtxt":
        dt.write1dArray(file, file_name)
    elif type[0:3] == "txt":
        dt.write(file, file_name)
    elif type == "scoredict":
        dt.save_csv_from_dict(file[0], file[1], file_name)
    elif type == "scoredictarray":
        dt.save_averages_and_final_csv(file[0], file[1], file[2], file[3], file_name)
    elif type == "csv":
        dt.write_csv(file_name, file[0], file[1], file[2])
    elif type == "gensim_save_model":
        file.save(file_name)
    elif type == "dct":
        dt.save_dict(file, file_name)
    else:
        raise ValueError("File type not recognized")
