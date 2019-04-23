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
dir1 = np.load("../../data/processed/movies/directions/fil/num_stw_num_stw_50_MDS_ndcg_1000_20000_0_dir.npy").transpose()
dir2 = np.load("../../data/processed/movies/directions/fil/num_stw_num_stw_50_PCA_ndcg_2000_20000_0_dir.npy").transpose()
dir3 = np.load("../../data/processed/movies/directions/fil/num_stw_num_stw_50_AWVEmp_ndcg_2000_20000_0_dir.npy").transpose()
words1 = np.load("../../data/processed/movies/rank/fil/num_stw_num_stw_50_MDS_ndcg_1000_20000_0_words.npy")
words2 = np.load("../../data/processed/movies/rank/fil/num_stw_num_stw_50_PCA_ndcg_2000_20000_0_words.npy")
words3 = np.load("../../data/processed/movies/rank/fil/num_stw_num_stw_50_AWVEmp_ndcg_2000_20000_0_words.npy")
words_to_get_amt = 1000

file_name = "comparing_mds_awv_fixed"
words_array = np.asarray([words1[:words_to_get_amt], words3[:words_to_get_amt]])
dir_array =np.asarray([dir1[:words_to_get_amt], dir3[:words_to_get_amt]])

import os
getDiff = False
remaining_inds = []
words_with_context = []
ctx_path = "../../data/processed/movies/vis/words_with_ctx "+file_name+".npy"
if os.path.exists(ctx_path) is True:
    words_with_context = np.load(ctx_path)
else:
    pairs = getPairs(*words_array)
    for i in range(len(pairs)):
        remaining_inds.append(termsUniqueOnlyTo(*pairs[i]))
    for i in range(len(remaining_inds)):
        word_arrays = contextualizeWords(words_array[i][remaining_inds[i]], dir_array[i][remaining_inds[i]], words_array[i], dir_array[i])
        words_with_context.append(word_arrays)
        print("---")
    np.save(ctx_path, words_with_context)

common_path = "../../data/processed/movies/vis/common_words "+file_name+".npy"
if os.path.exists(common_path) is True:
    common_words_ctx = np.load(common_path)
else:
    all_array = np.concatenate(words_array)
    all_dirs = np.concatenate(dir_array)
    ids = termsCommonTo(all_array.tolist(), len(words_array))
    print(ids)
    common_words_ctx = contextualizeWords(all_array[ids], all_dirs[ids],  all_array, all_dirs)
    np.save(common_path, common_words_ctx)
common_words_ctx_concat = np.concatenate(common_words_ctx)

print("---")
matching_concept_ids = []
for i in range(len(words_with_context)):
    matching_concept_ids.append([])
    for j in range(len(words_with_context[i])):
        for z in range(len(words_with_context[i][j])):
            for k in range(len(common_words_ctx_concat)):
                if words_with_context[i][j][z] == common_words_ctx_concat[k]:
                    matching_concept_ids[i].append(j)
                    break
        print(i, "/", len(words_with_context), j, "/", len(words_with_context[i]))

true_uniques = []
for i in range(len(matching_concept_ids)):
    true_uniques.append(np.delete(words_with_context[i], matching_concept_ids[i], axis=0))

for i in range(len(matching_concept_ids)):
    matching_concept_ids[i] = np.unique(matching_concept_ids[i])
fake_uniques = []
for i in range(len(matching_concept_ids)):
    fake_uniques.append(np.asarray(words_with_context[i])[matching_concept_ids[i]])

print("true uniques")
for i in range(len(true_uniques)):
    for j in range(len(true_uniques[i])):
        printPretty(true_uniques[i][j])
    print("-----")

print("fake uniques")
for i in range(len(fake_uniques)):
    for j in range(len(fake_uniques[i])):
        printPretty(fake_uniques[i][j])
    print("-----")
