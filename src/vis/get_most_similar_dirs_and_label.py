import numpy as np

from util import sim
from util import io
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
            sys.stdout.write(str(word_array[k]) + " (")
        elif k != len(word_array) - 1:
            sys.stdout.write(str(word_array[k]) + ", ")
        else:
            sys.stdout.write(str(word_array[k]))
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

"""
data_type = "placetypes"
fns = ["num_stw_num_stw_50_AWVEmp_ndcg_2000_10000_0_",
       "num_stw_US_200_Activ_tanh_Dropout_0.5_Hsize_[1000, 100]_BS_10_mlnrep_ndcg_2000_10000_0_",
       "num_stw_US_100_Activ_tanh_Dropout_0.25_Hsize_2_BS_10_mlnrep_ndcg_2000_10000_0_"]
"""
"""
data_type = "movies"
fns = ["num_stw_num_stw_200_MDS_ndcg_2000_10000_0_",
       "num_stw_US_20_Activ_tanh_Dropout_0.5_Hsize_[1000, 100]_mlnrep_ndcg_2000_10000_0_",
       "num_stw_US_300_Activ_tanh_Dropout_0.25_Hsize_3_mlnrep_ndcg_2000_10000_0_"]
"""
"""
data_type = "newsgroups"
fns = ["num_stw_num_stw_50_D2V_ndcg_2000_10000_0_",
       "num_stw_US_5_Activ_tanh_Dropout_0.5_Hsize_[1000, 100]_mlnrep_ndcg_2000_10000_0_",
       "num_stw_US_200_Activ_tanh_Dropout_0.1_Hsize_3_mlnrep_ndcg_2000_10000_0_"]
"""



"""
data_type = "placetypes"
fns = ["num_stw_num_stw_50_AWVEmp_10000_0_",
       "num_stw_US_100_Activ_tanh_Dropout_0.25_Hsize_2_BS_10_mlnrep_10000_0_",
       "num_stw_US_200_Activ_tanh_Dropout_0.5_Hsize_[1000, 100]_BS_10_mlnrep_10000_0_"]
"""
"""
data_type = "movies"
fns = ["num_stw_num_stw_200_MDS_10000_0_",
       "num_stw_US_20_Activ_tanh_Dropout_0.5_Hsize_[1000, 100]_mlnrep_10000_0_",
       "num_stw_US_300_Activ_tanh_Dropout_0.25_Hsize_3_mlnrep_10000_0_"]
"""
"""
data_type = "newsgroups"
fns = ["num_stw_num_stw_50_D2V_10000_0_",
       "num_stw_US_5_Activ_tanh_Dropout_0.5_Hsize_[1000, 100]_mlnrep_10000_0_",
       "num_stw_US_200_Activ_tanh_Dropout_0.1_Hsize_3_mlnrep_10000_0_"]
"""
data_type = "movies"
fns = ["num_stw_US_20_Activ_tanh_Dropout_0.5_Hsize_[1000, 100]_mlnrep_10000_0_"]

orig_fn = "../../data_paper\experimental results/chapter 5/"+data_type+"/all_dir/"
# Get the best-scoring terms and directions for each score-type for single directions
orig_space_dir = np.load(orig_fn+ fns[0] +"dir.npy")
#bow_space_dir = np.load(orig_fn+ fns[1] + "dir.npy")
#vector_space_dir = np.load(orig_fn+ fns[2]+ "dir.npy")

orig_space_words = np.load(orig_fn+ fns[0]+ "words.npy")
#bow_space_words = np.load(orig_fn+ fns[1] + "words.npy")
#vector_space_words = np.load(orig_fn+ fns[2] +"words.npy")

if len(orig_space_words) != len(orig_space_dir):# or len(bow_space_dir) != len(bow_space_words) or len(vector_space_dir) != len(vector_space_words):
    print("doesn't match", len(orig_space_words))#,len(orig_space_dir),len(bow_space_dir),len(bow_space_words),len(vector_space_dir),len(vector_space_words))
    orig_space_dir = orig_space_dir.transpose()
    if len(orig_space_words) != len(orig_space_dir):
        exit()

cutoff = len(orig_space_dir)

all_score_words = [orig_space_words[:cutoff] ]
all_score_dirs = [orig_space_dir[:cutoff]]

#ctx = np.load(orig_fn + fns[1] + "words_ctx.npy")
"""
for i in range(len(ctx)):
    print(ctx[i])
"""
all_score_words_ctx = []
for i in range(len(all_score_words)):
    word_arrays = contextualizeWords(all_score_words[i], all_score_dirs[i], all_score_words[i], all_score_dirs[i])
    all_score_words_ctx.append(word_arrays)
    print(i, "/", len(all_score_words)-1)

for i in range(len(all_score_words_ctx)):
    np.save(orig_fn + fns[i] + "words_ctx.npy", all_score_words_ctx[i])

for i in range(len(all_score_words_ctx)):
    for j in range(len(all_score_words_ctx[i])):
        printPretty(all_score_words_ctx[i][j])

    print("---")
    print("---")
    print("---")
    print("---")
    print("---")
