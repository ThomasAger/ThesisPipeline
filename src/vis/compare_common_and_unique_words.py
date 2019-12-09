import numpy as np
from util import io, vis as u
import os

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

data_type_1 = "placetypes"
first_fn = "num_stw_num_stw_50_AWVEmp_ndcg_2000_10000_0_"
second_fn = "num_stw_US_200_Activ_tanh_Dropout_0.5_Hsize_[1000, 100]_BS_10_mlnrep_ndcg_2000_10000_0_"
third_fn = "num_stw_US_100_Activ_tanh_Dropout_0.25_Hsize_2_BS_10_mlnrep_ndcg_2000_10000_0_"
csv_1 = io.read_csv("../../data_paper\experimental results\chapter 5/"+data_type_1+"/csv/num_stw_num_stw_50_AWVEmp_10000_0Streamed_ndcg.csv")
csv_2 = io.read_csv("../../data_paper\experimental results\chapter 5/"+data_type_1+"/csv/num_stw_US_200_Activ_tanh_Dropout_0.5_Hsize_[1000, 100]_BS_10_mlnrep_10000_0Streamed_ndcg.csv")
csv_3 = io.read_csv("../../data_paper\experimental results\chapter 5/"+data_type_1+"/csv/num_stw_US_100_Activ_tanh_Dropout_0.25_Hsize_2_BS_10_mlnrep_10000_0Streamed_ndcg.csv")
words_to_get_amt = 500

"""
data_type_1 = "movies"
first_fn = "num_stw_num_stw_200_MDS_ndcg_2000_10000_0_"
second_fn = "num_stw_US_20_Activ_tanh_Dropout_0.5_Hsize_[1000, 100]_mlnrep_ndcg_2000_10000_0_"
third_fn = "num_stw_US_300_Activ_tanh_Dropout_0.25_Hsize_3_mlnrep_ndcg_2000_10000_0_"
csv_1 = io.read_csv("../../data_paper\experimental results\chapter 5/"+data_type_1+"/csv/num_stw_num_stw_200_MDS_10000_0_ndcg.csv")
csv_2 = io.read_csv("../../data_paper\experimental results\chapter 5/"+data_type_1+"/csv/num_stw_US_20_Activ_tanh_Dropout_0.5_Hsize_[1000, 100]_mlnrep_10000_0_ndcg.csv")
csv_3 = io.read_csv("../../data_paper\experimental results\chapter 5/"+data_type_1+"/csv/num_stw_US_300_Activ_tanh_Dropout_0.25_Hsize_3_mlnrep_10000_0_ndcg.csv")
words_to_get_amt = 2000
"""
"""
data_type_1 = "newsgroups"
first_fn = "num_stw_num_stw_50_D2V_ndcg_2000_10000_0_"
second_fn = "num_stw_US_5_Activ_tanh_Dropout_0.5_Hsize_[1000, 100]_mlnrep_ndcg_2000_10000_0_"
third_fn = "num_stw_US_200_Activ_tanh_Dropout_0.1_Hsize_3_mlnrep_ndcg_2000_10000_0_"
csv_1 = io.read_csv("../../data_paper\experimental results\chapter 5/"+data_type_1+"/csv/num_stw_num_stw_50_D2V_10000_0Streamed_ndcg.csv")
csv_2 = io.read_csv("../../data_paper\experimental results\chapter 5/"+data_type_1+"/csv/num_stw_US_5_Activ_tanh_Dropout_0.5_Hsize_[1000, 100]_mlnrep_10000_0Streamed_ndcg.csv")
csv_3 = io.read_csv("../../data_paper\experimental results\chapter 5/"+data_type_1+"/csv/num_stw_US_200_Activ_tanh_Dropout_0.1_Hsize_3_mlnrep_10000_0Streamed_ndcg.csv")
words_to_get_amt = 2000
"""
csvs = [csv_1, csv_2, csv_3]
fns = [first_fn, second_fn, third_fn]
dir1 = np.load("../../data_paper\experimental results\chapter 5/"+data_type_1+"/dir/"+first_fn+"dir.npy").transpose()
dir2 = np.load("../../data_paper\experimental results\chapter 5/"+data_type_1+"/dir/"+second_fn+"dir.npy").transpose()
dir3 = np.load("../../data_paper\experimental results\chapter 5/"+data_type_1+"/dir/"+third_fn+"dir.npy").transpose()
words1 = io.import1dArray("../../data_paper\experimental results\chapter 5/"+data_type_1+"/dir/"+first_fn+"words.txt")
words2 = io.import1dArray("../../data_paper\experimental results\chapter 5/"+data_type_1+"/dir/"+second_fn+"words.txt")
words3 = io.import1dArray("../../data_paper\experimental results\chapter 5/"+data_type_1+"/dir/"+third_fn+"words.txt")
ctx1 = np.load("../../data_paper\experimental results\chapter 5/"+data_type_1+"/dir/"+first_fn+"words_ctx.npy")
ctx2 = np.load("../../data_paper\experimental results\chapter 5/"+data_type_1+"/dir/"+second_fn+"words_ctx.npy")
ctx3 = np.load("../../data_paper\experimental results\chapter 5/"+data_type_1+"/dir/"+third_fn+"words_ctx.npy")

unique1 = "../../data_paper\experimental results\chapter 5/"+data_type_1+"/dir/"+first_fn+"unique.txt"
unique2 = "../../data_paper\experimental results\chapter 5/"+data_type_1+"/dir/"+second_fn+"unique.txt"
unique3 = "../../data_paper\experimental results\chapter 5/"+data_type_1+"/dir/"+third_fn+"unique.txt"
unique_fns = [unique1, unique2, unique3]
common_fn = "../../data_paper\experimental results\chapter 5/"+data_type_1+"/dir/common.txt"
if words2[999] == words3[999]:
    raise ValueError("imported same words")

if len(dir1) != len(words1) or len(dir2) != len(words2) or len(dir3) != len(words3):
    print("Len doesnt match")
    exit()
else:
    print("len matches")
file_name = data_type_1 + "unsupervised_vector_bow"
words_array = [words1[:words_to_get_amt], words2[:words_to_get_amt], words3[:words_to_get_amt]]
dir_array =[dir1[:words_to_get_amt], dir2[:words_to_get_amt], dir3[:words_to_get_amt]]
ctx_array = [ctx1, ctx2, ctx3]
ctx_path = "../../data/processed/"+data_type_1+"/vis/unique_words_ctx_" + file_name + ".npy"
if True:

    getDiff = False
    remaining_inds = []
    unique_terms = []
    words_with_context = []
    if False:#os.path.exists(ctx_path) is True:
        words_with_context = np.load(ctx_path, allow_pickle=True)
    else:
        pairs = u.getPairs(*words_array) # Get pairs of words_1:words_2+words_3
        for i in range(len(pairs)):
            remaining_inds.append(u.termsUniqueOnlyTo(*pairs[i])) # Delete any terms that occurred in all arrays
            unique_terms.append(words_array[i][remaining_inds[i]])
        for i in range(len(remaining_inds)):
            print(i)
            word_arrays = u.mapWordsToContext(words_array[i][remaining_inds[i]],ctx_array[i])
            words_with_context.append(word_arrays) # Contextualize the resulting words
            print("---")
        for i in range(len(words_with_context)):
            np.save(ctx_path+fns[i]+".npy", words_with_context[i])
        np.save(ctx_path, words_with_context)
    common_words = None
    common_path = "../../data/processed/"+data_type_1+"/vis/common_words_"+file_name+".npy"
    ctx_concat_fn = "../../data/processed/"+data_type_1+"/vis/ctx_concat"+file_name+".npy"

    all_array = np.concatenate(words_array)
    #all_dirs = np.concatenate(dir_array)
    ids = u.termsCommonTo(all_array.tolist(), len(words_array))
    common_words = np.asarray(all_array[ids])
    __unused, common_ids = np.unique(common_words, return_index=True)
    common_words = common_words[common_ids.argsort()]
    #print(ids)
    common_words_ctx_concat = u.mapWordsToContext(common_words, np.concatenate(ctx_array))
    # = contextualizeWords(all_array[ids], all_dirs[ids],  all_array, all_dirs)
    np.save(common_path, common_words)
    np.save(ctx_concat_fn, common_words_ctx_concat)


    all_words_context_in_one_array = []
    for i in range(len(common_words_ctx_concat)):
        for j in range(len(common_words_ctx_concat[i])):
            all_words_context_in_one_array.append(common_words_ctx_concat[i][j])

    print("---")
    matching_concept_ids = []
    for i in range(len(words_with_context)):
        matching_concept_ids.append([])
        for j in range(len(words_with_context[i])):
            for z in range(len(words_with_context[i][j])):
                for k in range(len(all_words_context_in_one_array)):
                    if words_with_context[i][j][z] == all_words_context_in_one_array[k]:
                        matching_concept_ids[i].append(j)
                        break
            print(i, "/", len(words_with_context), j, "/", len(words_with_context[i]))




    true_uniques = []
    for i in range(len(matching_concept_ids)):
        true_uniques.append(np.delete(unique_terms[i], matching_concept_ids[i], axis=0))



    np.save(unique_fns[0][:-4] + "_all.npy", unique_terms)
    np.save(unique_fns[0][:-4] + "_true_all.npy", true_uniques)
    np.save(common_fn[:-4] + "_all.npy", common_words)

else:
    words_with_context = np.load(ctx_path, allow_pickle=True)
    unique_terms = np.load(unique_fns[0][:-4] + "_all.npy", allow_pickle=True)
    true_uniques =np.load(unique_fns[0][:-4] + "_true_all.npy", allow_pickle=True)
    common_words = np.load(common_fn[:-4] + "_all.npy", allow_pickle=True)

scores = []
names = []
for i in range(len(csvs)):
    scores.append(csvs[i].iloc[:,0].values)
    names.append(csvs[i].index.values)

for i in range(len(scores)):
    ids = np.asarray(scores[i]).argsort()
    names[i] = names[i][ids][::-1]



unique_terms_ctx = []
for i in range(len(unique_terms)):
    sorted_terms = u.sortByScoreWordArrays(unique_terms[i], names[i])
    unique_terms_ctx.append(u.mapWordsToContext(sorted_terms, ctx_array[i]))
    a = u.getPretty(unique_terms_ctx[i])
    io.write1dArray(a, unique_fns[i])

true_uniques_ctx = []
for i in range(len(true_uniques)):
    sorted_terms = u.sortByScoreWordArrays(true_uniques[i], names[i])
    true_uniques_ctx.append(u.mapWordsToContext(sorted_terms, ctx_array[i]))
    a = u.getPretty(true_uniques_ctx[i])
    io.write1dArray(a, unique_fns[i][:-4] + "_true.txt")

sorted_terms = u.sortByScoreWordArrays(common_words, names[0])
common_words_ctx = u.mapWordsToContext(sorted_terms, np.concatenate(ctx_array))
a = u.getPretty(common_words_ctx)
io.write1dArray(a, common_fn)
"""
for i in range(len(words_with_context)):
    save = getPretty(words_with_context[i])
    io.write1dArray(save, unique_fns[i])

io.write1dArray(getPretty(common_words_ctx_concat), common_fn)

for i in range(len(matching_concept_ids)):
    matching_concept_ids[i] = np.unique(matching_concept_ids[i])
fake_uniques = []
for i in range(len(matching_concept_ids)):
    print(matching_concept_ids[i])
    fake_uniques.append(np.asarray(words_with_context[i])[matching_concept_ids[i]])
"""


"""
print("fake uniques")
for i in range(len(fake_uniques)):
    for j in range(len(fake_uniques[i])):
        printPretty(fake_uniques[i][j])
    print("-----")
"""