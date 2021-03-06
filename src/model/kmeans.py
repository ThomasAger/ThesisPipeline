from common.SaveLoadPOPO import SaveLoadPOPO
from common.Method import Method
from sklearn.cluster import KMeans
import numpy as np
from util.save_load import SaveLoad
class KMeansCluster(Method):
    cluster_dirs = None
    cluster_ranks = None
    cluster_names = None
    cluster_amt = None
    folder_name = None
    feature_names = None
    centroids = None

    def __init__(self, features, cluster_amt, file_name, folder_name, save_class, feature_names):
        self.folder_name = folder_name
        self.cluster_amt = cluster_amt
        self.features = features
        self.feature_names = feature_names

        super().__init__(file_name, save_class)

    def makePopos(self):
        self.cluster_dirs = SaveLoadPOPO(self.cluster_dirs, self.folder_name + "directions/"+ self.file_name + "_all.npy", "npy")
        self.centroids = SaveLoadPOPO(self.centroids, self.folder_name + "directions/"+ self.file_name + "_cent.npy", "npy")
        self.cluster_names = SaveLoadPOPO(self.cluster_names, self.folder_name + "names/" +self.file_name + ".txt", "1dtxts")

    def makePopoArray(self):
        self.popo_array = [self.cluster_dirs, self.cluster_names, self.centroids]

    def process(self):
        ms = KMeans(verbose=1, n_clusters=self.cluster_amt)
        ms.fit(self.features)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        cluster_names = []
        for i in range(len(cluster_centers)):
            cluster_names.append("")
        for i in range(len(self.feature_names)):
            cluster_names[labels[i]] += self.feature_names[i] + " "
        self.cluster_names.value = cluster_names
        self.centroids.value = cluster_centers

        self.cluster_dirs.value = []
        for i in range(len(self.feature_names)):
            self.cluster_dirs.value.append([])

        for i in range(len(self.feature_names)):
            self.cluster_dirs.value[labels[i]].append(self.features[i])

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



