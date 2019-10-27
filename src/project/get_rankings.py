from common import Method
from common.SaveLoadPOPO import SaveLoadPOPO
from util.save_load import SaveLoad
import numpy as np
from model import svm

import scipy.sparse as sp
import os
class GetRankings(Method.Method):
    word_dir = None
    words = None
    output_directions = None
    output_folder = None
    directions = None
    dirs = None
    bowmin = None
    bowmax = None
    new_word_dict = None
    space = None
    LR = None

    def __init__(self, dirs, space,  words, save_class, file_name, output_folder, bowmin, bowmax):
        self.words = words
        self.output_folder = output_folder
        self.space = space
        self.dirs = dirs
        self.bowmin = bowmin
        self.bowmax = bowmax
        super().__init__(file_name, save_class)

    def makePopos(self):

        self.rankings = SaveLoadPOPO(np.empty(shape = (len(self.words.keys()), len(self.space))), self.output_folder + "rank/" + self.file_name + "_" + str(self.bowmin) + "_" + str(self.bowmax) + "_rank.npy", "npy")
        # If the rankings for this situation exists, then dont load the overall.
        self.rank_dir = SaveLoadPOPO({}, self.output_folder + "rank/" + self.file_name + "_rank_dir.npy", "npy_dict")
        if self.save_class.exists([self.rank_dir]) and self.save_class.exists([self.rankings]) is False:
            self.rank_dir.value = self.save_class.load(self.rank_dir)
            print("pred_dir loaded successfully")

    def makePopoArray(self):
        self.popo_array = [self.rankings, self.rank_dir]

    def get_dp(self, direction):
        ranking = np.empty(len(self.space))
        for i in range(len(self.space)):
            ranking[i] = np.dot(direction, self.space[i])
        return ranking

    def process(self):
        i = 0
        for key, value in self.words.items():
            # If the word isn't in the saved version
            if key not in self.rank_dir.value:
                self.rank_dir.value[key] = self.get_dp(self.dirs[value])
            self.rankings.value[value] = self.rank_dir.value[key]
            print(i, "/", len(self.words.keys()), key)
            i += 1
        super().process()

    def getRankings(self):
        if self.processed is False:
            self.rankings.value = self.save_class.load(self.rankings)
        return self.rankings.value

class GetRankingsNoSave(Method.Method):
    word_dir = None
    words = None
    output_directions = None
    output_folder = None
    directions = None
    dirs = None
    bowmin = None
    bowmax = None
    new_word_dict = None
    space = None
    LR = None

    def __init__(self, dirs, space,  words, save_class, file_name, output_folder, bowmin, bowmax):
        self.words = words
        self.output_folder = output_folder
        self.space = space
        self.dirs = dirs
        self.bowmin = bowmin
        self.bowmax = bowmax
        super().__init__(file_name, save_class)

    def makePopos(self):

        self.rankings = SaveLoadPOPO(np.empty(shape = (len(self.words.keys()), len(self.space))), self.output_folder + "rank/" + self.file_name + "_" + str(self.bowmin) + "_" + str(self.bowmax) + "_rank.npy", "npy")

    def makePopoArray(self):
        self.popo_array = [self.rankings]

    def get_dp(self, direction):
        ranking = np.empty(len(self.space))
        for i in range(len(self.space)):
            ranking[i] = np.dot(direction, self.space[i])
        return ranking

    def process(self):
        i = 0
        for key, value in self.words.items():
            self.rankings.value[value] = self.get_dp(self.dirs[value])
            #print(i, "/", len(self.words.keys()), key)
            i += 1
        super().process()

    def getRankings(self):
        if self.processed is False:
            self.rankings.value = self.save_class.load(self.rankings)
        return self.rankings.value


class GetRankingsStreamed(Method.Method):
    word_dir = None
    words = None
    output_directions = None
    output_folder = None
    directions = None
    dirs = None
    bowmin = None
    bowmax = None
    new_word_dict = None
    space = None
    LR = None

    def __init__(self, dirs, space,  words, save_class, file_name, output_folder, bowmin, bowmax):
        self.words = words
        self.output_folder = output_folder
        self.space = space
        self.dirs = dirs
        self.bowmin = bowmin
        self.bowmax = bowmax
        super().__init__(file_name, save_class)
        self.rank_fn = self.output_folder + "rank/" + self.file_name + "_" + str(self.bowmin) + "_" + str(self.bowmax) + "_rank.txt"

    def get_dp(self, direction):
        ranking = np.empty(len(self.space))
        for i in range(len(self.space)):
            ranking[i] = np.dot(direction, self.space[i])
        return ranking

    def process(self):
        j = 0
        keys = np.asarray(list(self.words.keys()))
        vals = np.asarray(list(self.words.values()))
        sorted_ids = np.argsort(vals)
        sorted_keys = keys[sorted_ids]
        sorted_vals = vals[sorted_ids]
        with open(self.rank_fn, 'w') as write_file:
            prev_sorted_val = 0
            for i in range(len(sorted_vals)):
                if (sorted_vals[i] - prev_sorted_val) > 1:
                    raise ValueError("Dict has missing value, cannot write in rows")
                ranks = self.get_dp(self.dirs[sorted_vals[i]])
                rank_str = ""
                for k in range(len(ranks)):
                    rank_str += str(ranks[k]) + " "
                write_file.write(rank_str + "\n")
                print(j, "/", len(self.words.keys()), sorted_keys[j])
                j += 1
                prev_sorted_val = sorted_vals[i]
        super().process()

    def getRankings(self):
        return self.rank_fn


    def process_and_save(self):
        if os.path.exists(self.rank_fn) is False or self.save_class.rewrite is True:
            if self.save_class.rewrite is True:
                print("Rewriting")
            print("doesnt exist", self.rank_fn)
            self.process()
        else:
            print("Exists", self.rank_fn)



if __name__ == '__main__':
    np.load("E:\PhD\Code\ThesisPipeline\ThesisPipeline\data\processed\placetypes\directions\words/num_stw_50_new_wdct_NB_69_NA_1313.npy", allow_pickle=True)
    print("kay")