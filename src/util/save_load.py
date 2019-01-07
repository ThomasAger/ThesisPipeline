
from gensim.corpora import Dictionary
import os
import numpy as np
import util.io as dt
import scipy.sparse as sp
import gensim.models as g
class SaveLoad:
    rewrite = None
    no_save = None

    def __init__(self, rewrite=False, no_save=False):
        self.rewrite = rewrite
        self.no_save = no_save

    def load(self, popo_array):
        for i in range(len(popo_array)):
            if popo_array[i].file_name is None or popo_array[i].file_type is None:
                raise ValueError("None type" + popo_array[i].value.file_name)
            popo_array[i].value = load_by_type(popo_array[i].file_type, popo_array[i].file_name)
        return popo_array

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
            print(popo_array[0].file_name, "Rewriting...")
            return False
        all_exist = 0
        for i in range(len(popo_array)):
            if os.path.exists(popo_array[i].file_name):
                print(popo_array[i].file_name, "Already exists")
                all_exist += 1
            else:
                print(popo_array[i].file_name, "Doesn't exist")
        if all_exist == len(popo_array):
            return True
        return False


def load_by_type(type, file_name):
    if type == "npy":
        file = np.load(file_name)
    elif type == "scipy":
        file = sp.load_npz(file_name)
    elif type == "dct":
        file = dt.load_dict(file_name)
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
    else:
        raise ValueError("File type not recognized")
    return file

def save_by_type(file, type, file_name):
    if type == "npy":
        np.save(file_name, file)
    elif type == "scipy":
        sp.save_npz(file_name, file)
    elif type == "gensim":
        file.save(file_name)
    elif type == "dct":
        dt.save_dict(file, file_name)
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
    else:
        raise ValueError("File type not recognized")
