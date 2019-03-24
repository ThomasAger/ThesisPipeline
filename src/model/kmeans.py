from common.SaveLoadPOPO import SaveLoadPOPO
from common.Method import Method
from sklearn.cluster import KMeans
class MethodName(Method):
    cluster_dirs = None
    cluster_ranks = None
    cluster_names = None
    cluster_amt = None
    folder_name = None

    def __init__(self, features, cluster_amt, file_name, folder_name, save_class):
        self.folder_name = folder_name
        self.cluster_amt = cluster_amt
        self.features = features
        self.features = self.features[:100]

        super().__init__(file_name, save_class)


    def makePopos(self):
        self.cluster_dirs = SaveLoadPOPO(self.cluster_dirs, self.folder_name + self.file_name + "_" + str(self.cluster_amt) + ".npy", "npy")
        self.cluster_names = SaveLoadPOPO(self.cluster_names, "filename", "npy")

    def makePopoArray(self):
        self.popo_array = []

    def process(self):
        # Process
        super().process()

    def process(self):
        ms = KMeans(verbose=1, n_clusters=self.cluster_amt)
        ms.fit(self.features)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        self.cluster_names.value = labels
        self.cluster_dirs.value = cluster_centers

