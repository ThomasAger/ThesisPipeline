import numpy as np
from util import io as dt

def getTopEntitiesOnRanking(ranking, entity_names, cluster_names, cluster_length=3, top_x=-1, cluster_ids=None, output=True):
    if cluster_ids is not None:
        ranking = ranking[cluster_ids]
        cluster_names = cluster_names[cluster_ids]
    for i in range(len(cluster_names)):
        cluster_names[i] = cluster_names[i]
    top_entities = []
    top_rankings = []
    for c in range(len(ranking)):
        top_cluster_entities = []
        top_cluster_rankings = []
        sorted_cluster = np.asarray(list(reversed(entity_names[np.argsort(ranking[c])])))
        sorted_rankings = np.asarray(list(reversed(ranking[c][np.argsort(ranking[c])])))
        for e in range(len(sorted_cluster)):
            top_cluster_entities.append(sorted_cluster[e])
            top_cluster_rankings.append(sorted_rankings[e])
            if e == top_x:
                break
        top_entities.append(top_cluster_entities)
        top_rankings.append(top_cluster_rankings)
        if output:
            print("Cluster:", cluster_names[c],  "Entites", top_cluster_entities)
            #print("Cluster:", cluster_names[c],  "Entites", top_cluster_rankings)
    return top_entities, top_rankings

data_type = "movies"
orig_fn = "../../data/processed/" + data_type + "/"
file_name = "num_stw_num_stw_50_MDS_1397_13279_"
ranking = np.load(orig_fn + "rank/" + file_name + "rank.npy")
entity_names = dt.import1dArray(orig_fn + "corpus/entity_names.txt")
cluster_names = np.load(orig_fn + "directions/bow/" + file_name + "words.npy")
getTopEntitiesOnRanking(ranking, entity_names, cluster_names, top_x=3)