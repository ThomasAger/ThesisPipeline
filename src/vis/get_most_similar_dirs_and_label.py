import numpy as np

from util import sim
def getPairs(*params):
    pairs = []
    for i in range(len(params)):
        pair = []
        pair.append(params[i])
        concat_final = np.asarray([])
        for j in range(len(params)):
            if j == i:
                continue
            else:
                concat_final = np.concatenate([concat_final, params[j]], axis=0)
        pair.append(concat_final)
        pairs.append(pair)
    return pairs

def termsUniqueOnlyTo(this_array, all_other_arrays):
    ids_to_del = []
    for i in range(len(this_array)):
        for j in range(len(all_other_arrays)):
            if this_array[i] == all_other_arrays[j]:
                ids_to_del.append(i)
    inds = list(range(len(this_array)))
    remaining_inds = np.delete(inds, ids_to_del)
    return remaining_inds

def termsCommonTo(arrays, amt_of_arrays):
    ids_to_del = []
    counts = []
    for i in range(len(arrays)):
        counts.append(arrays.count(arrays[i]))
    common_ids = []
    for i in range(len(counts)):
        if counts[i] == amt_of_arrays:
            common_ids.append(i)
    common_ids = np.unique(common_ids)
    return common_ids

import sys

def printPretty(word_array):
    for k in range(len(word_array)):
        if k == 0:
            sys.stdout.write(word_array[k] + " (")
        elif k != len(word_array) - 1:
            sys.stdout.write(word_array[k] + ", ")
        else:
            sys.stdout.write(word_array[k])
    sys.stdout.write(")\n")
def contextualizeWords(words, word_directions, context_words, ctx_word_directions):
    word_arrays = []
    for i in range(len(words)):
        inds = sim.getXMostSimilarIndex(word_directions[i], ctx_word_directions, [], 2)
        word_arrays.append([words[i]])
        for j in inds:
            word_arrays[i].append(context_words[j])
        printPretty(word_arrays[i])
    return word_arrays
# Get the best-scoring terms and directions for each score-type for single directions
sent_dir = np.load("../../data/processed/sentiment/directions/fil/num_stw_num_stw_100_D2V_ndcg_1000_10000_0_dir.npy").transpose()
news_dir = np.load("../../data/processed/newsgroups/directions/fil/num_stw_num_stw_50_D2V_ndcg_2000_10000_0_dir.npy").transpose()
placetype_dir = np.load("../../data/processed/placetypes/directions/fil/num_stw_num_stw_50_PCA_kappa_1000_10000_0_dir.npy").transpose()
reut_dir = np.load("../../data/processed/reuters/directions/fil/num_stw_num_stw_200_MDS_ndcg_2000_5000_0_dir.npy").transpose()

sent_words = np.load("../../data/processed/sentiment/rank/fil/num_stw_num_stw_100_D2V_ndcg_1000_10000_0_words.npy")
news_words = np.load("../../data/processed/newsgroups/rank/fil/num_stw_num_stw_50_D2V_ndcg_2000_10000_0_words.npy")
placetype_words = np.load("../../data/processed/placetypes/rank/fil/num_stw_num_stw_50_PCA_kappa_1000_10000_0_words.npy")
reut_words = np.load("../../data/processed/reuters/rank/fil/num_stw_num_stw_200_MDS_ndcg_2000_5000_0_words.npy")

all_score_words = [sent_words, news_words, placetype_words, reut_words]
all_score_dirs = [sent_dir, news_dir, placetype_dir, reut_dir]
words_to_get_amt = 1000

file_name = "top_scoring"
all_score_words_ctx = []
for i in range(len(all_score_words)):
    word_arrays = contextualizeWords(all_score_words[i], all_score_dirs[i], all_score_words[i], all_score_dirs[i])
    all_score_words_ctx.append(word_arrays)
    print(i, "/", len(all_score_words)-1)

for i in range(len(all_score_words_ctx)):
    for j in range(len(all_score_words_ctx[i])):
        printPretty(all_score_words_ctx[i][j])
    print("---")
    print("---")
    print("---")
    print("---")
    print("---")
