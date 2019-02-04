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

    def __init__(self, rankings,  words, save_class, file_name, output_folder, bowmin, bowmax):
        self.rankings = rankings
        self.output_folder = output_folder
        super().__init__(file_name, save_class)

    def makePopos(self):

        self.ndcg = SaveLoadPOPO(np.empty(len(self.rankings), dtype=object), self.output_folder + "rank/" + self.file_name + "_" + str(self.bowmin) + "_" + str(self.bowmax) + "_rank.npy", ".npy")


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
            self.rankings.value[i] = self.rank_dir.value(self.words[i])
        super().process()

    def getRankings(self):
        if self.processed is False:
            self.rankings.value = self.save_class.load(self.rankings)
        return self.rankings.value


def ranking_precision_score(y_true, y_score, k=10):
    """Precision at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    precision @k : float
    """
    unique_y = np.unique(y_true)

    if len(unique_y) > 2:
        raise ValueError("Only supported for two relevance levels.")

    pos_label = unique_y[1]
    n_pos = np.sum(y_true == pos_label)

    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    n_relevant = np.sum(y_true == pos_label)

    # Divide by min(n_pos, k) such that the best achievable score is always 1.0.
    return float(n_relevant) / min(n_pos, k)


def average_precision_score(y_true, y_score, k=10):
    """Average precision at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    average precision @k : float
    """
    unique_y = np.unique(y_true)

    if len(unique_y) > 2:
        raise ValueError("Only supported for two relevance levels.")

    pos_label = unique_y[1]
    n_pos = np.sum(y_true == pos_label)

    order = np.argsort(y_score)[::-1][:min(n_pos, k)]
    y_true = np.asarray(y_true)[order]

    score = 0
    for i in range(len(y_true)):
        if y_true[i] == pos_label:
            # Compute precision up to document i
            # i.e, percentage of relevant documents up to document i.
            prec = 0
            for j in range(0, i + 1):
                if y_true[j] == pos_label:
                    prec += 1.0
            prec /= (i + 1.0)
            score += prec

    if n_pos == 0:
        return 0

    return score / n_pos


def dcg_score(y_true, y_score, k=10, gains="exponential"):
    """Discounted cumulative gain (DCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    DCG @k : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    if gains == "exponential":
        gains = 2 ** y_true - 1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10, gains="exponential"):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    NDCG @k : float
    """
    best = dcg_score(y_true, y_true, k, gains)
    actual = dcg_score(y_true, y_score, k, gains)
    return actual / best


# Alternative API.

def dcg_from_ranking(y_true, ranking):
    """Discounted cumulative gain (DCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    ranking : array-like, shape = [k]
        Document indices, i.e.,
            ranking[0] is the index of top-ranked document,
            ranking[1] is the index of second-ranked document,
            ...
    k : int
        Rank.
    Returns
    -------
    DCG @k : float
    """
    y_true = np.asarray(y_true)
    ranking = np.asarray(ranking)
    rel = y_true[ranking]
    gains = 2 ** rel - 1
    discounts = np.log2(np.arange(len(ranking)) + 2)
    overall = gains / discounts
    sum = np.sum(overall)
    return sum


def ndcg_from_ranking(y_true, ranking):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    ranking : array-like, shape = [k]
        Document indices, i.e.,
            ranking[0] is the index of top-ranked document,
            ranking[1] is the index of second-ranked document,
            ...
    k : int
        Rank.
    Returns
    -------
    NDCG @k : float
    """
    k = len(ranking)
    best_ranking = np.argsort(y_true)[::-1]
    best = dcg_from_ranking(y_true, best_ranking[:k])
    dcg = dcg_from_ranking(y_true, ranking)
    return dcg / best


def getNDCG(rankings_fn, fn, data_type, bow_fn, ppmi_fn, lowest_count, rewrite_files=False, highest_count = 0, classification = ""):

    # Check if the NDCG scores have already been calculated, if they have then skip.


    # Get the file names for the PPMI values for every word and a list of words ("names")

    # Process the rankings and the PPMI line-by-line so as to not run out of memory
    ndcg_a = []
    #spearman_a = []
    with open(rankings_fn) as rankings:
        r = 0
        for lr in rankings:
                for lp in ppmi:
                    # Get the plain-number ranking of the rankings, e.g. "1, 4, 3, 50"
                    sorted_indices = np.argsort(list(map(float, lr.strip().split())))[::-1]
                    # Convert PPMI scores to floats
                    # Get the NDCG score for the PPMI score, which is a valuation, compared to the indice of the rank
                    ndcg = ndcg_from_ranking(lp, sorted_indices)

                    # Add to array and print
                    ndcg_a.append(ndcg)
                    print("ndcg", ndcg, names[r], r)
                    """
                    smr = spearmanr(ppmi_indices, sorted_indices)[1]
                    spearman_a.append(smr)
                    print("spearman", smr, names[r], r)
                    """
                    r+=1
                    break
    # Save NDCG
    dt.write1dArray(ndcg_a, ndcg_fn)
    #dt.write1dArray(spearman_a, spearman_fn)



class GetNDCG(Method.Method):
    word_dir = None
    words = None
    output_directions = None
    output_folder = None
    directions = None
    ranks = None
    bowmin = None
    bowmax = None
    new_word_dict = None
    ppmi = None
    LR = None
    ppmi_id_dct = None

    def __init__(self, ranks, ppmi, words, ppmi_id_dct, save_class, file_name, output_folder, bowmin, bowmax):
        self.words = words
        self.output_folder = output_folder
        self.ppmi = ppmi
        self.ranks = ranks
        self.bowmin = bowmin
        self.bowmax = bowmax
        self.ppmi_id_dct = ppmi_id_dct
        super().__init__(file_name, save_class)

    def makePopos(self):
        self.ndcg_scores = SaveLoadPOPO(np.empty(len(self.ranks), dtype=object),
                                        self.output_folder  + self.file_name + "_" + str(
                                            self.bowmin) + "_" + str(self.bowmax) + "_ndcg.npy", "npy")
        # If the rankings for this situation exists, then dont load the overall.
        self.ndcg_dir = SaveLoadPOPO({}, self.output_folder + self.file_name + "_ndcg_dir.npy", "npy_dict")
        if self.save_class.exists([self.ndcg_dir]) and self.save_class.exists([self.ndcg_scores]) is False:
            self.ndcg_dir.value = self.save_class.load(self.ndcg_dir)
            print("pred_dir loaded successfully")
        self.csv_data = SaveLoadPOPO([["ndcg"],[],[self.words]],
                                        self.output_folder + "csv/"+ self.file_name + "_" + str(
                                            self.bowmin) + "_" + str(self.bowmax) + "_ndcg.csv","csv")


    def makePopoArray(self):
        self.popo_array = [self.ndcg_scores, self.ndcg_dir, self.csv_data]

    def process(self):
        for i in range(len(self.words)):
            if self.words[i] not in self.ndcg_dir.value:
                sorted_indices = np.flipud(np.argsort(self.ranks[i]))
                # Get the NDCG score for the PPMI score, which is a valuation, compared to the indice of the rank
                ndcg = ndcg_from_ranking(np.asarray(self.ppmi[self.ppmi_id_dct[self.words[i]]].todense())[0], sorted_indices)
                self.ndcg_dir.value[self.words[i]] = ndcg
            self.ndcg_scores.value[i] = self.ndcg_dir.value[self.words[i]]
            print(i, "/", len(self.words), self.words[i], self.ndcg_dir.value[self.words[i]])#
        self.csv_data.value[1] = [self.ndcg_scores.value]
        super().process()

    def getNDCG(self):
        if self.processed is False:
            self.ndcg_scores.value = self.save_class.load(self.ndcg_scores)
        return self.ndcg_scores.value


if __name__ == '__main__':
    print("kay")