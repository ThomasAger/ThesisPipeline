from util import io as dt
import os
class SaveLoad:
    rewrite = None

    def __init__(self, rewrite=False):
        self.rewrite = rewrite

    def load(self, popo_array):
        for i in range(len(popo_array)):
            if popo_array[i].file_name is None or popo_array[i].file_type is None:
                raise ValueError("None type" + popo_array[i].value.file_name)
            popo_array[i].value = dt.load_by_type(popo_array[i].file_type, popo_array[i].file_name)
        return popo_array

    def save(self, popo_array):
        for i in range(len(popo_array)):
            if popo_array[i].value is None or popo_array[i].file_name is None or popo_array[i].file_type is None:
                raise ValueError("None type" + popo_array[i].value.file_name)
            dt.save_by_type(popo_array[i].value, popo_array[i].file_type, popo_array[i].file_name)

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