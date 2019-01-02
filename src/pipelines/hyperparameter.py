from pipelines import derrac
from itertools import product


def main():
    ### GENERAL
    data_type="newsgroups"
    split_ids=None
    all_x = []
    all_y = []

    # If using the development set or the test set
    dev_set = [True]
    cross_val = 5

    ### DIRECTIONS

    # The minimum amount of documents the word must occur
    min_freq = [200]
    # The frequency minus this number is the max amount of documents the word can occur
    max_freq = [10]

    ### SCORING

    # The method of scoring the directions used in the later clustering algorithm
    score_type = ["kappa", "ndcg", "acc"]

    ### NNET PARAMS

    # The amount of epochs the finetuning network is ran for
    epochs = [300]

    ### CLUSTER PARAMS

    # The clustering algorithm to use, kmeans/meanshift are the scikit-learn implementations
    cluster_type = ["kmeans"]  # "derrac", "meanshift"
    # The share that word-vectors have when averaged with the directions, e.g. 1=all word vectors, 0=no word vectors
    word_vectors = [0.5]

    ## derrac
    # The amount of clusters
    cluster_centers = [200]
    # The amount of directions used to form the cluster centers
    cluster_center_directions = [400]
    # The amount of directions clustered with the centers
    cluster_directions = [2000]

    ## meanshift
    # The parameter for the distance between points, uses estimate bandwidth and then modifies it
    bandwidth = [1]  # amount estimate bandwidth is multiplied by

    ## kmeans
    # Number of time the k-means algorithm will be run with different centroid seeds.
    # The final results will be the best output of n_init consecutive runs in terms of inertia.
    n_init = [10]
    # Maximum number of iterations of the k-means algorithm for a single run.
    max_iter = [300]

    var_names = ["min_freq", "max_freq", "score_type", "epochs", "cluster_type", "word_vectors", "cluster_centers", "cluster_center_directions",
            "cluster_directions", "bandwidth", "n_init", "max_iter"]

    all_params = list(
        product(
            min_freq, max_freq, score_type, epochs, cluster_type, word_vectors, cluster_centers, cluster_center_directions,
            cluster_directions, bandwidth, n_init, max_iter
        )
    )

    all_p = []
    for i in range(len(all_params)):
        pdict = {}
        for j in range(len(var_names)):
            pdict[var_names[j]] = all_params[i][j]
        all_p.append(pdict)


    space = None
    bow = None

    all_scores = {}
    for i in range(len(all_p)):
        p = all_p[i]
        all_scores[i] = derrac.pipeline(data_type=data_type, all_x=p["all_x"], all_y=p["all_y"], min_freq=p["min_freq"], max_freq=p["max_freq"], score_type=p["score_type"]
                                                 , epochs=p["epochs"], cluster_type=p["cluster_type"], word_vectors=p["word_vectors"]
                                                 , cluster_center_directions=p["cluster_center_directions"], cluster_directions=["cluster_directions"],
                                                 bandwidth=p["bandwidth"], n_init=p["n_init"] , max_iter=p["max_iter"],
                                                 cluster_centers=p["cluster_centers"])



if __name__ == '__main__':
    main()