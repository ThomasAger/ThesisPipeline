from common.SaveLoadPOPO import SaveLoadPOPO
from common.Method import Method
from sklearn import preprocessing
import numpy as np
class ConsolidateClasses(Method):
    centroids = None
    bow = None
    clusters = None
    cluster_names = None
    output_folder = None
    multidim_array = None
    normalized_output = None

    def __init__(self, token2id, bow, clusters, cluster_names, file_name, output_folder, save_class):

        self.output_folder = output_folder
        self.token2id = token2id
        self.bow = bow
        self.clusters = clusters
        self.cluster_names = cluster_names
        super().__init__(file_name, save_class)

    def makePopos(self):
        self.centroids = SaveLoadPOPO(self.centroids, self.output_folder + self.file_name + ".npy", "npy")

    def makePopoArray(self):
        self.popo_array = [self.centroids]

    def process(self):
        # Process
        cluster_bows = []
        for i in range(len(self.cluster_names)):
            names = self.cluster_names[i].split()
            bow_names = []
            for j in range(len(names)):
                names[j] = names[j].strip()
                bow_name = self.bow[self.token2id[names[j]]]
                bow_names.append(np.asarray(bow_name.todense())[0])

            cluster_bow = np.zeros(len(bow_names[0]), dtype=np.int32)
            for j in range(len(bow_names)):
                for k in range(len(bow_names[j])):
                    if bow_names[j][k] >= 1:
                        cluster_bow[k] = 1
            cluster_bows.append(cluster_bow)

        self.centroids.value = cluster_bows
        print("completed")
        super().process()

    def getCentroids(self):
        if self.processed is False:
            self.centroids.value = self.save_class.load(self.centroids)
        return self.centroids.value