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

    def __init__(self, features, cluster_amt, file_name, folder_name, save_class, feature_names, top_dir_amt):
        self.folder_name = folder_name
        self.cluster_amt = cluster_amt
        self.features = features
        self.feature_names = feature_names
        self.top_dir_amt = top_dir_amt * cluster_amt

        super().__init__(file_name, save_class)

    def makePopos(self):
        self.cluster_dirs = SaveLoadPOPO(self.cluster_dirs, self.folder_name + "directions/"+ self.file_name + ".npy", "npy")
        self.cluster_names = SaveLoadPOPO(self.cluster_names, self.folder_name + "names/" +self.file_name + ".npy", "npy")

    def makePopoArray(self):
        self.popo_array = [self.cluster_dirs, self.cluster_names]

    def process(self):

        centroid_ids = [0]

        for c in range(self.cluster_amt):
            max_values = []
            for i in range(self.top_dir_amt):
                cos_vals = []
                for j in range(len(centroid_ids)):
                     cos_vals.append(cosSim(self.features[i], self.features[centroid_ids[j]]))
                max_values.append(np.max(cos_vals))
            min_val_id = py.aminId(max_values)
            centroid_ids.append(min_val_id)
            print(c, "/", self.cluster_amt, self.feature_names[min_val_id], max_values[min_val_id])



    def getClusters(self):
        if self.processed is False:
            self.cluster_dirs.value = self.save_class.load(self.cluster_dirs)
        return self.cluster_dirs.value

    def getClusterNames(self):
        if self.processed is False:
            self.cluster_names.value = self.save_class.load(self.cluster_names)
        return self.cluster_names.value



