from io import io
import numpy as np

from scipy.stats import spearmanr
from io import io

import numpy as np
from scipy.stats import spearmanr


def get_spearman(array1, array2):
    spearman = np.zeros(len(array1))
    for i in range(len(array1)):
        correlation, pvalue = spearmanr(array1[i], array2[i])
        spearman[i] = correlation
        print(i, len(array1), spearman[i])
    return spearman


if __name__ == '__main__':
    data_type = "reuters"
    orig_fn = "../../data/"+data_type+"/"
    norm_fn = "2-all_mds200CV1S0 SFT0 allL0100.95 LR kappa KMeans CA400 MC1 MS0.4 ATS1000 DS800"
    rank = np.load(orig_fn + "rank/numeric/"+norm_fn+".npy")
    target_rank = np.load(orig_fn + "finetune/boc/"+norm_fn+"FT BOCFi.npy")
    spearman = get_spearman(rank, target_rank)
    cluster_names = io.import1dArray(orig_fn + "cluster/dict/" + norm_fn + ".txt")



    csv = io.get_CSV_from_arrays([cluster_names, spearman], ["cluster_names", "spearman"])

    io.write_string(csv, orig_fn + "spearman.csv")