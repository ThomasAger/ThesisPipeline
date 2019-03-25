from common import Method
from common.SaveLoadPOPO import SaveLoadPOPO
import numpy as np
class GetTopScoringDirs(Method.Method):
    score_ind = None
    top_scoring_dir = None
    directions = None
    new_word2id_dict = None
    output_folder = None
    dir = None
    words = None

    def __init__(self, file_name, save_class, output_folder, score_ind, top_scoring_dir, directions, new_word2id_dict):
        self.score_ind = score_ind
        self.top_scoring_dir = top_scoring_dir
        self.directions = directions
        self.new_word2id_dict = new_word2id_dict
        self.output_folder = output_folder
        super().__init__(file_name, save_class)

    def makePopos(self):
        self.dir = SaveLoadPOPO([],self.output_folder + "fil/" + self.file_name  + "_dir.npy", "npy")

    def makePopoArray(self):
        self.popo_array = [self.dir]

    def process(self):
        inds = np.flipud(np.argsort(self.score_ind))[:self.top_scoring_dir]
        self.dir.value = self.directions[inds].transpose()
        super().process()

    def getDir(self):
        if self.processed is False:
            self.dir.value = self.save_class.load(self.dir)
        return self.dir.value


class GetTopScoringDirsStreamed(Method.Method):
    score_ind = None
    top_scoring_dir = None
    dir_fn = None
    new_word2id_dict = None
    output_folder = None
    dir = None
    words = None

    def __init__(self, file_name, save_class, output_folder, score_ind, top_scoring_dir, dir_fn, new_word2id_dict):
        self.score_ind = score_ind
        self.top_scoring_dir = top_scoring_dir
        self.dir_fn = dir_fn
        self.new_word2id_dict = new_word2id_dict
        self.output_folder = output_folder
        super().__init__(file_name, save_class)

    def makePopos(self):
        self.dir = SaveLoadPOPO([], self.output_folder + "fil/" + self.file_name  + "_dir.npy", "npy")

    def makePopoArray(self):
        self.popo_array = [self.dir]

    def process(self):
        inds = np.flipud(np.argsort(self.score_ind))[:self.top_scoring_dir]
        i = 0
        word_len = len(self.new_word2id_dict.keys())
        with open(self.dir_fn) as infile:
            for line in infile:
                if i in inds:
                    split = line.split()
                    float_split = np.empty(len(split), dtype=np.float64)
                    for j in range(len(split)):
                        float_split[j] = float(split[j])
                    self.dir.value.append(float_split)
                i += 1
                print(i, "/", word_len)
        self.dir.value = np.asarray(self.dir.value)
        self.dir.value = self.dir.value.transpose()
        super().process()

    def getRank(self):
        if self.processed is False:
            self.dir.value = self.save_class.load(self.dir)
        return self.dir.value



