import numpy as np
from util import io
import pydotplus
from util import py
from util import vis
data_types = ["movies","placetypes"]
embedding_type = ["unsupervised", "bow", "vector","unsupervised", "bow", "vector"]

cluster_fn1 = "../../data_paper\experimental results\chapter 5/"+data_types[0]+"/csv/num_stw_num_stw_200_MDS_10000_0_ndcg.csv"
cluster_fn2 = "../../data_paper\experimental results\chapter 5/"+data_types[0]+"/csv/num_stw_US_20_Activ_tanh_Dropout_0.5_Hsize_[1000, 100]_mlnrep_10000_0_ndcg.csv"
cluster_fn3 = "../../data_paper\experimental results\chapter 5/"+data_types[0]+"/csv/num_stw_US_300_Activ_tanh_Dropout_0.25_Hsize_3_mlnrep_10000_0_ndcg.csv"
cluster_fn4 = "../../data_paper\experimental results\chapter 5/"+data_types[1]+"/csv/num_stw_num_stw_50_AWVEmp_10000_0Streamed_ndcg.csv"
cluster_fn5 = "../../data_paper\experimental results\chapter 5/"+data_types[1]+"/csv/num_stw_US_200_Activ_tanh_Dropout_0.5_Hsize_[1000, 100]_BS_10_mlnrep_10000_0Streamed_ndcg.csv"
cluster_fn6 = "../../data_paper\experimental results\chapter 5/"+data_types[1]+"/csv/num_stw_US_100_Activ_tanh_Dropout_0.25_Hsize_2_BS_10_mlnrep_10000_0Streamed_ndcg.csv"

rank_fn1 = "../../data_paper\experimental results\chapter 5/"+data_types[0]+"/dir/" + "num_stw_num_stw_200_MDS_ndcg_2000_10000_0_"
rank_fn2 = "../../data_paper\experimental results\chapter 5/"+data_types[0]+"/dir/" + "num_stw_US_20_Activ_tanh_Dropout_0.5_Hsize_[1000, 100]_mlnrep_ndcg_2000_10000_0_"
rank_fn3 = "../../data_paper\experimental results\chapter 5/"+data_types[0]+"/dir/" + "num_stw_US_300_Activ_tanh_Dropout_0.25_Hsize_3_mlnrep_ndcg_2000_10000_0_"
rank_fn4 = "../../data_paper\experimental results\chapter 5/"+data_types[1]+"/dir/" + "num_stw_num_stw_50_AWVEmp_ndcg_2000_10000_0_"
rank_fn5 = "../../data_paper\experimental results\chapter 5/"+data_types[1]+"/dir/" + "num_stw_US_200_Activ_tanh_Dropout_0.5_Hsize_[1000, 100]_BS_10_mlnrep_ndcg_2000_10000_0_"
rank_fn6 = "../../data_paper\experimental results\chapter 5/"+data_types[1]+"/dir/" + "num_stw_US_100_Activ_tanh_Dropout_0.25_Hsize_2_BS_10_mlnrep_ndcg_2000_10000_0_"

entity_fn1 = io.import1dArray("../../data_paper\experimental results\chapter 5/"+data_types[0]+"/entity_names.txt")
entity_fn2 = io.import1dArray("../../data_paper\experimental results\chapter 5/"+data_types[1]+"/entity_names.txt")
entities = [[entity_fn1,entity_fn1,entity_fn1],[entity_fn2,entity_fn2,entity_fn2]]
word_fns = [[cluster_fn1, cluster_fn2, cluster_fn3], [cluster_fn4, cluster_fn5, cluster_fn6]]
rank_fns = [[rank_fn1,rank_fn2,rank_fn3],[rank_fn4,rank_fn5,rank_fn6]]
ctx_1 = "../../data_paper\experimental results\chapter 5/"+data_types[0]+"/all_dir/num_stw_num_stw_200_MDS_10000_0_"
ctx_2 = "../../data_paper\experimental results\chapter 5/"+data_types[0]+"/all_dir/num_stw_US_20_Activ_tanh_Dropout_0.5_Hsize_[1000, 100]_mlnrep_10000_0_"
ctx_3 = "../../data_paper\experimental results\chapter 5/"+data_types[0]+"/all_dir/num_stw_US_300_Activ_tanh_Dropout_0.25_Hsize_3_mlnrep_10000_0_"
ctx_4 = "../../data_paper\experimental results\chapter 5/"+data_types[1]+"/all_dir/num_stw_num_stw_50_AWVEmp_10000_0_"
ctx_5 = "../../data_paper\experimental results\chapter 5/"+data_types[1]+"/all_dir/num_stw_US_200_Activ_tanh_Dropout_0.5_Hsize_[1000, 100]_BS_10_mlnrep_10000_0_"
ctx_6 = "../../data_paper\experimental results\chapter 5/"+data_types[1]+"/all_dir/num_stw_US_100_Activ_tanh_Dropout_0.25_Hsize_2_BS_10_mlnrep_10000_0_"
ctxs = [[ctx_1, ctx_2, ctx_3], [ctx_4, ctx_5, ctx_6]]
common_fns = ["all_tree_labels_trimmed_100","all_tree_labels_trimmed_100","all_tree_labels_trimmed_100","all_tree_labels_trimmed_100","all_tree_labels_trimmed_100","all_tree_labels_trimmed_100"]
for z in range(len(data_types)):

    print(z)
    common_sorted = io.import1dArray("../../data_paper\experimental results\chapter 5/" + data_types[z] + "/dir/common.txt")
    common = []
    for i in range(len(common_sorted)):
        common.append(common_sorted[i].split()[0].strip())
    print("worked")
    for g in range(len(rank_fns[z])):
        print(g)
        top_words = io.import1dArray(rank_fns[z][g] + "words.txt")
        all_words = io.read_csv(word_fns[z][g], dtype=str).index.values
        all_words = vis.sortByScoreWordArrays(top_words, all_words)
        ranks = np.load(rank_fns[z][g] +"rank.npy")
        if len(ranks) != len(all_words):
            ranks = ranks.transpose()

        if len(all_words) != len(ranks):
            print("Didn't match all words")
            exit()
            break
        ids = []
        for j in range(len(common)):
            for i in range(len(all_words)):
                if all_words[i] == common[j]:
                    ids.append(i)
                    break

        print("got ids", "matched", len(ids), "out of", len(common))
        ranks_to_get = ranks[ids]
        all_top_entities = []
        top_5_entities =[]
        for i in range(len(common)):
            try:
                all_top_entities.append(np.asarray(list(reversed(entities[z][g][np.argsort(ranks_to_get[i])]))))
                top_5_entities.append(vis.clusterPretty(all_top_entities[i][:5]))
            except IndexError:
                print("L")
        print("got entities")
        ctx_words = vis.getPretty(vis.mapWordsToContext(common, np.load(  ctxs[z][g] + "words_ctx.npy")))
        print("k?")
        io.write_csv(rank_fns[z][g]+"dir_top_5_entities.csv",
                     ["ranks"],
                     [top_5_entities], key=ctx_words)

