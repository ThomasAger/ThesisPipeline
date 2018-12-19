from io import io

import numpy as np
import scipy.sparse as sp
from spearman_difference import get_spearman

import model.tree as tree
from score import classify
from util import split


# For each cluster, take each word and count the number of objects where the words appear once
def countObjects(bow, words, names_dict):
    inds = []
    for word_list in words:
        ind = []
        for w in word_list:
            ind.append(names_dict[w])
        inds.append(ind)

    counts = []

    for i in range(len(inds)):
        count = np.zeros(len(inds[i]), dtype=np.int)
        for j in range(len(inds[i])):
            for k in range(len(bow)):
                if bow[k][inds[i][j]] > 0:
                    count[j] += 1
        counts.append(count)

    return counts



if __name__ == '__main__':
    data_type = "reuters"
    orig_fn = "../../data/"+data_type+"/"
    ft_fn = "2-all_mds200CV1S0 SFT0 allL0100.95 LR kappa KMeans CA400 MC1 MS0.4 ATS1000 DS800FT BOCFi NT[200]tanh300V1.2"
    norm_fn = "2-all_mds200CV1S0 SFT0 allL0100.95 LR kappa KMeans CA400 MC1 MS0.4 ATS1000 DS800"
    rank = np.load(orig_fn + "rank/numeric/"+norm_fn+".npy").transpose()
    classes = io.import2dArray(orig_fn + "/classify/topics/class-all", "i")
    class_names = io.import1dArray(orig_fn + "/classify/topics/names.txt")

    x_train, y_train, x_test, y_test, x_dev, y_dev = split.split(rank, classes, "reuters")

    y_train = y_train.transpose()
    y_test = y_test.transpose()
    class_names = class_names

    d_t = tree.DecisionTree(x_train, y_train, x_test)

    pred = d_t.get_predictions(max_depth=1, criterion="entropy", class_weight="balanced")

    score_dict = classify.Score(y_test, pred, class_names).calculate(verbose=True)

    clusters = np.load(orig_fn + "cluster/clusters/" + norm_fn + ".npy")
    cluster_names = io.import1dArray(orig_fn + "cluster/dict/" + norm_fn + ".txt")

    final_clusters, final_rankings, final_fns, fn_ids = d_t.getNodesToDepth(clusters, cluster_names)

    ids = np.zeros(len(final_fns), dtype=np.int)

    for j in range(len(final_fns)):
        for i in range(len(cluster_names)):
            if cluster_names[i] == final_fns[j][0]:
                ids[j] = i
                break

    final_fn_words = []
    for i in range(len(final_fns)):
        split = final_fns[i][0].split()[:3]
        split[0] = split[0][:-1]
        final_fn_words.append(split)
        print(split)

    target_rank = np.load(orig_fn + "finetune/boc/" + norm_fn + "FT BOCFi.npy")
    spearman = get_spearman(rank.transpose()[ids], target_rank[ids])

    bow = sp.load_npz(orig_fn + "bow/frequency/phrases/simple_numeric_stopwords_bow 10-0.95-all.npz")
    bow_names = io.import1dArray(orig_fn + "bow/names/simple_numeric_stopwords_words 10-0.95-all.txt")
    bow = np.asarray(bow.transpose().todense())

    names_dict = {}

    for i in range(len(bow_names)):
        names_dict[bow_names[i]] = i

    split.check_shape(bow, data_type)
    split.check_features(bow)

    counts = countObjects(bow, final_fn_words, names_dict)

    csv = io.get_CSV_from_arrays([class_names, final_fn_words, counts, spearman], ["class names", "top cluster", "amt of objects", "spearman"])

    io.write_string(csv, orig_fn + "experiment_results.csv")