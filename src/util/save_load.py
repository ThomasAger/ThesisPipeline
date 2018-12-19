from util import io as dt
import numpy as np
import scipy.sparse as sp
from gensim.Corpora import Dictionary
import os
class SaveLoad:

    def load(self, popo_array):
        for i in range(len(popo_array)):
            popo_array.value = dt.load_by_type(popo_array[i].type, popo_array[i].file_name)
        return popo_array

    def save(self, popo_array):
        for i in range(len(popo_array)):
            popo_array.value = dt.save_by_type(popo_array[i].value, popo_array[i].type, popo_array[i].file_name)

    def exists(self, popo_array):
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