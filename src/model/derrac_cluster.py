from common.SaveLoadPOPO import SaveLoadPOPO
from common.Method import Method
from sklearn.cluster import KMeans
import numpy as np
from util.save_load import SaveLoad
from scipy import spatial
from numpy import dot
from numpy.linalg import norm
from util import py
def cosSim(a, b):
    return dot(a, b)/(norm(a)*norm(b))
# The distance is 1 - the cosine similarity
def cosDis(a, b):
    return 1- dot(a, b)/(norm(a)*norm(b))

class DerracCluster(Method):
    cluster_dirs = None
    cluster_ranks = None
    cluster_names = None
    cluster_amt = None
    folder_name = None
    feature_names = None
    top_dir_amt = None
    centroids = None

    def __init__(self, features, cluster_amt, file_name, folder_name, save_class, feature_names, top_dir_amt):

        self.folder_name = folder_name
        self.cluster_amt = cluster_amt
        self.features = features
        self.feature_names = feature_names.tolist()
        self.top_dir_amt = top_dir_amt * cluster_amt
        print(self.top_dir_amt, "top dir")
        super().__init__(file_name, save_class)

    def makePopos(self):
        self.cluster_dirs = SaveLoadPOPO(self.cluster_dirs, self.folder_name + "directions/"+ self.file_name + "_all.npy", "npy")
        self.centroids = SaveLoadPOPO(self.centroids, self.folder_name + "directions/"+ self.file_name + "_cent.npy", "npy")
        self.cluster_names = SaveLoadPOPO(self.cluster_names, self.folder_name + "names/" +self.file_name + ".txt", "1dtxts")

    def makePopoArray(self):
        self.popo_array = [self.cluster_dirs, self.cluster_names, self.centroids]

    def process(self):

        centroid_ids = [0]


        print(1, "/", self.cluster_amt, self.feature_names[0])


        if self.top_dir_amt > len(self.features):
            self.top_dir_amt = len(self.features)

        if self.top_dir_amt == self.cluster_amt:
            centroid_ids = range(self.cluster_amt)
        else:
            for c in range(1, self.cluster_amt): # For each cluster
                max_values = []
                for i in range(self.top_dir_amt): # For each word direction
                    cos_vals = []
                    for j in range(len(centroid_ids)): # Get the similarity between the word direction and each of the cluster directions
                         cos_vals.append(cosSim(self.features[i], self.features[centroid_ids[j]]))
                    max_values.append(np.max(cos_vals)) # Add the maximum similarity among the clusters  to max_values
                min_val_id = py.aminId(max_values) # Get the minimum
                centroid_ids.append(min_val_id)
                print(c+1, "/", self.cluster_amt, self.feature_names[min_val_id], max_values[min_val_id])

        self.centroids.value = self.features[centroid_ids]
        self.cluster_dirs.value = self.features[centroid_ids]
        self.cluster_names.value = np.asarray(self.feature_names)[centroid_ids]
        self.cluster_names.value = self.cluster_names.value.tolist()
        self.features = np.delete(self.features, centroid_ids, axis=0)
        self.feature_names = np.delete(self.feature_names, centroid_ids, axis=0)

        print(len(self.centroids.value))

        for i in range(len(self.features)):
            cluster_sims = []
            for c in range(self.cluster_amt):
                cluster_sims.append(cosSim(self.features[i], self.centroids.value[c]))
            best_cluster_id = py.amaxId(cluster_sims)
            self.centroids.value[best_cluster_id] = np.average([self.centroids.value[best_cluster_id], self.features[i]], axis=0)
            self.cluster_dirs.value[best_cluster_id] = self.features[i]
            self.cluster_names.value[best_cluster_id] += " " + self.feature_names[i]
            print(i, "/", len(self.features), self.cluster_names.value[best_cluster_id])

        print("-------COMPLETE-------")
        for i in range(len(self.cluster_names.value)):
            print(self.cluster_names.value[i])


    def getDirName(self):
        return self.centroids.file_name

    def getClusters(self):
        if self.processed is False:
            self.cluster_dirs.value = self.save_class.load(self.cluster_dirs)
        return self.cluster_dirs.value

    def getCentroids(self):
        if self.processed is False:
            self.centroids.value = self.save_class.load(self.centroids)
        return self.centroids.value

    def getClusterNames(self):
        if self.processed is False:
            self.cluster_names.value = self.save_class.load(self.cluster_names)
        return self.cluster_names.value



