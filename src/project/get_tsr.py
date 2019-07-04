from common import Method
from common.SaveLoadPOPO import SaveLoadPOPO
import numpy as np
class GetTopScoringRanks(Method.Method):
    score_ind = None
    top_scoring_dir = None
    rankings = None
    new_word2id_dict = None
    output_folder = None
    rank = None
    words = None

    def __init__(self, file_name, save_class, output_folder, score_ind, top_scoring_dir, rankings, new_word2id_dict):
        self.score_ind = score_ind
        self.top_scoring_dir = top_scoring_dir
        self.rankings = rankings
        self.new_word2id_dict = new_word2id_dict
        self.output_folder = output_folder
        super().__init__(file_name, save_class)

    def makePopos(self):
        self.words = SaveLoadPOPO(self.words,
                                  self.output_folder + "fil/" + self.file_name + "_words.npy", "npy")
        self.rank = SaveLoadPOPO(self.rank,
                                 self.output_folder + "fil/" + self.file_name  + "_rank.npy", "npy")

    def makePopoArray(self):
        self.popo_array = [self.words, self.rank]

    def process(self):
        inds = np.flipud(np.argsort(self.score_ind))[:self.top_scoring_dir]
        self.rank.value = self.rankings[inds].transpose()
        self.words.value = np.asarray(list(self.new_word2id_dict.keys()))[inds]
        super().process()

    def getRank(self):
        if self.processed is False:
            self.rank.value = self.save_class.load(self.rank)
        return self.rank.value

    def getWords(self):
        if self.processed is False:
            self.words.value = self.save_class.load(self.words)
        return self.words.value

class GetTopScoringRanksStreamed(Method.Method):
    score_ind = None
    top_scoring_dir = None
    rank_fn = None
    new_word2id_dict = None
    output_folder = None
    rank = None
    words = None

    def __init__(self, file_name, save_class, output_folder, score_ind, top_scoring_dir, rank_fn, new_word2id_dict):
        self.score_ind = score_ind
        self.top_scoring_dir = top_scoring_dir
        self.rank_fn = rank_fn
        self.new_word2id_dict = new_word2id_dict
        self.output_folder = output_folder
        super().__init__(file_name, save_class)

    def makePopos(self):
        self.words = SaveLoadPOPO([],
                                  self.output_folder + "fil/" + self.file_name + "_words.npy", "npy")
        self.rank = SaveLoadPOPO([],
                                 self.output_folder + "fil/" + self.file_name  + "_rank.npy", "npy")

    def makePopoArray(self):
        self.popo_array = [self.words, self.rank]

    def process(self):
        inds = np.flipud(np.argsort(self.score_ind))[:self.top_scoring_dir]
        i = 0
        internal_rank = []
        internal_mapping = {}
        amt_added = 0
        word_len = len(self.new_word2id_dict.keys())
        with open(self.rank_fn) as infile:
            for line in infile:
                for k in range(len(inds)):
                    if i == inds[k]:
                        print(k)
                        split = line.split()
                        float_split = np.empty(len(split), dtype=np.float64)
                        for j in range(len(split)):
                            float_split[j] = float(split[j])
                        internal_rank.append(float_split)
                        internal_mapping[inds[k]] = amt_added
                        amt_added += 1
                        break
                i += 1
                print(i, "/", word_len)
        final_rank_array = []
        for i in range(len(inds)):
            final_rank_array.append(internal_rank[internal_mapping[inds[i]]])
        self.rank.value = np.asarray(final_rank_array).transpose()
        self.words.value = np.asarray(list(self.new_word2id_dict.keys()))[inds]
        super().process()

    def getRank(self):
        if self.processed is False:
            self.rank.value = self.save_class.load(self.rank)
        return self.rank.value

    def getWords(self):
        if self.processed is False:
            self.words.value = self.save_class.load(self.words)
        return self.words.value

