import numpy as np
from util import io
import pydotplus
from util import py
from util import vis
data_types = ["placetypes","placetypes","placetypes","movies","movies","movies"]
embedding_type = ["unsupervised", "bow", "vector","unsupervised", "bow", "vector"]
cluster_fn1 = "../../data_paper\experimental results\chapter 5/"+data_types[0]+"/cluster/" + "num_stw_num_stw_50_PCA_kappa_1000_10000_0_rank_5_300_1e-05_k-means++_100_kmeans.txt"
cluster_fn2 = "../../data_paper\experimental results\chapter 5/"+data_types[1]+"/cluster/" + "200_Activ_tanh_Dropout_0.5_Hsize_[1000, 100]_BS_10_mlnrep_10_1000_0.0_k-means++_200_kmeans.txt"
cluster_fn3 = "../../data_paper\experimental results\chapter 5/"+data_types[2]+"/cluster/" + "100_Activ_tanh_Dropout_0.25_Hsize_2_BS_10_mlnrep_5_300_0.0001_k-means++_50_kmeans.txt"
cluster_fn4 = "../../data_paper\experimental results\chapter 5/"+data_types[3]+"/cluster/" + "num_stw_num_stw_200_MDS_ndcg_1000_10000_0_rank_50_100_0.0_k-means++_200_kmeans.txt"
cluster_fn5 = "../../data_paper\experimental results\chapter 5/"+data_types[4]+"/cluster/" + "20_Activ_tanh_Dropout_0.5_Hsize_[1000, 100]_mlnrep_10_1000_0.0_k-means++_200_kmeans.txt"
cluster_fn6 = "../../data_paper\experimental results\chapter 5/"+data_types[5]+"/cluster/" + "300_Activ_tanh_Dropout_0.25_Hsize_3_mlnrep_50_1000_0.0_k-means++_200_kmeans.txt"
rank_fn1 = "_best_100_rank.npy"
rank_fn2 = "_best_200_rank.npy"
rank_fn3 = "_best_50_rank.npy"
rank_fn4 = "_best_200_rank.npy"
rank_fn5 = "_best_200_rank.npy"
rank_fn6 = "_best_200_rank.npy"
entity_fn1 = io.import1dArray("../../data_paper\experimental results\chapter 5/"+data_types[0]+"/entity_names.txt")
entity_fn2 = io.import1dArray("../../data_paper\experimental results\chapter 5/"+data_types[5]+"/entity_names.txt")
entities = [entity_fn1,entity_fn1,entity_fn1,entity_fn2,entity_fn2,entity_fn2]
cluster_fns = [cluster_fn1, cluster_fn2, cluster_fn3, cluster_fn4, cluster_fn5, cluster_fn6]
rank_fns = [rank_fn1,rank_fn2,rank_fn3,rank_fn4,rank_fn5,rank_fn6]
label_fns = ["all_tree_labels_trimmed_100","all_tree_labels_trimmed_100","all_tree_labels_trimmed_100","all_tree_labels_trimmed_100","all_tree_labels_trimmed_100","all_tree_labels_trimmed_100"]
for z in range(len(data_types)):
    labels = np.load("../../data_paper/experimental results/chapter 5/"+data_types[z]+"/tree_labels/"+embedding_type[z]+"/"+label_fns[z]+".npy", allow_pickle=True)
    clusters = io.import1dArray(cluster_fns[z])
    ranks = np.load(cluster_fns[z][:-4] + rank_fns[z])
    final_labels = []

    for i in range(len(labels)):
        if len(labels[i]) == 0:
            print("Empty label")
            exit()
        final_labels.append(labels[i][-1].strip())
    ids = []
    for j in range(len(final_labels)):
        for i in range(len(clusters)):
            if clusters[i] == final_labels[j]:
                ids.append(i)
                break
    ranks_to_get = ranks[ids]

    all_top_entities = []
    for i in range(len(ranks_to_get)):
        all_top_entities.append(np.asarray(list(reversed(entities[z][np.argsort(ranks_to_get[i])]))))
        print(final_labels[i], all_top_entities[i][:5])
    np.save(cluster_fns[z][:-4] + "top_entities.npy",all_top_entities)
