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



def contextualizeWords(words, word_directions, context_words, ctx_word_directions):
    word_arrays = []
    for i in range(len(words)):
        inds = sim.getXMostSimilarIndex(word_directions[i], ctx_word_directions, [], 2)
        word_arrays.append([words[i]])
        for j in inds:
            word_arrays[i].append(context_words[j])
        print(word_arrays[i])
    return word_arrays
# Get the terms unique to only that space-type
"""
dir1 = np.load("../../data/processed/newsgroups/directions/fil/num_stw_num_stw_50_MDS_f1_1000_20000_0_dir.npy")[:1000]
dir2 = np.load("../../data/processed/newsgroups/directions/fil/num_stw_num_stw_50_AWVEmp_ndcg_2000_10000_0_dir.npy")[:1000]
dir3 = np.load("../../data/processed/newsgroups/directions/fil/num_stw_num_stw_50_PCA_kappa_1000_5000_0_dir.npy")[:1000]
dir4 = np.load("../../data/processed/newsgroups/directions/fil/num_stw_num_stw_50_D2V_ndcg_2000_10000_0_dir.npy")[:1000]
words1 = np.load("../../data/processed/newsgroups/rank/fil/num_stw_num_stw_50_MDS_f1_1000_20000_0_words.npy")[:1000]
words2 = np.load("../../data/processed/newsgroups/rank/fil/num_stw_num_stw_50_AWVEmp_ndcg_2000_10000_0_words.npy")[:1000]
words3 = np.load("../../data/processed/newsgroups/rank/fil/num_stw_num_stw_50_PCA_kappa_1000_5000_0_words.npy")[:1000]
words4 = np.load("../../data/processed/newsgroups/rank/fil/num_stw_num_stw_50_D2V_ndcg_2000_10000_0_words.npy")[:1000]
"""
dir1 = np.load("../../data/processed/newsgroups/directions/fil/num_stw_num_stw_50_MDS_ndcg_1000_20000_0_dir.npy").transpose()
dir2 = np.load("../../data/processed/newsgroups/directions/fil/num_stw_num_stw_50_AWVEmp_ndcg_2000_10000_0_dir.npy").transpose()
dir3 = np.load("../../data/processed/newsgroups/directions/fil/num_stw_num_stw_50_PCA_ndcg_1000_5000_0_dir.npy").transpose()
dir4 = np.load("../../data/processed/newsgroups/directions/fil/num_stw_num_stw_50_D2V_ndcg_2000_10000_0_dir.npy").transpose()
words1 = np.load("../../data/processed/newsgroups/rank/fil/num_stw_num_stw_50_MDS_ndcg_1000_20000_0_words.npy")
words2 = np.load("../../data/processed/newsgroups/rank/fil/num_stw_num_stw_50_AWVEmp_ndcg_2000_10000_0_words.npy")
words3 = np.load("../../data/processed/newsgroups/rank/fil/num_stw_num_stw_50_PCA_ndcg_1000_5000_0_words.npy")
words4 = np.load("../../data/processed/newsgroups/rank/fil/num_stw_num_stw_50_D2V_ndcg_2000_10000_0_words.npy")
words_to_get_amt = 500
words_array = [words1[:words_to_get_amt], words2[:words_to_get_amt], words3[:words_to_get_amt], words4[:words_to_get_amt]]
dir_array = [dir1[:words_to_get_amt], dir2[:words_to_get_amt], dir3[:words_to_get_amt], dir4[:words_to_get_amt]]
ctx_dir_array = [dir1[:1000], dir2[:1000], dir3[:1000], dir4[:1000]]
context_words_array = [words1[:1000], words2[:1000], words3[:1000], words4[:1000]]
pairs = getPairs(*words_array)
remaining_inds = []
for i in range(len(pairs)):
    remaining_inds.append(termsUniqueOnlyTo(*pairs[i]))
words_with_context = []
for i in range(len(remaining_inds)):
    word_arrays = contextualizeWords(words_array[i][remaining_inds[i]], dir_array[i][remaining_inds[i]], context_words_array[i], ctx_dir_array[i])
    words_with_context.append(word_arrays)
    print(word_arrays)
    print("---")