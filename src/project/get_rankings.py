from common import Method
from common.SaveLoadPOPO import SaveLoadPOPO
from util.save_load import SaveLoad
import numpy as np
from model import svm

import scipy.sparse as sp
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

        self.rankings = SaveLoadPOPO(np.empty(len(self.words), dtype=object), self.output_folder + "rank/" + self.file_name + "_" + str(self.bowmin) + "_" + str(self.bowmax) + "_rank.npy", "npy")
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
        for i in range(len(self.words)):
            if self.words[i] not in self.rank_dir.value:
                self.rank_dir.value[self.words[i]] = self.get_dp(self.dirs[i])
            self.rankings.value[i] = self.rank_dir.value[self.words[i]]
            print(i, "/", len(self.words), self.words[i])
        super().process()

    def getRankings(self):
        if self.processed is False:
            self.rankings.value = self.save_class.load(self.rankings)
        return self.rankings.value


if __name__ == '__main__':
    print("kay")