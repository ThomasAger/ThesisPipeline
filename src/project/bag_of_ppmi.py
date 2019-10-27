from common.SaveLoadPOPO import SaveLoadPOPO
from common.Method import Method
import numpy as np
from sklearn import preprocessing
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
class PAV(Method):
    bag_of_clusters = None
    pav_classes = None
    output_folder = None
    do_plot = None
    bow_names = None
    sparse_freqs = None
    ranking = None
    cluster_names = None

    def __init__(self, file_name, output_folder, bow_names, sparse_freqs, ranking, cluster_names, do_plot, save_class):

        self.output_folder = output_folder
        self.bow_names = bow_names
        self.sparse_freqs = sparse_freqs
        self.ranking = ranking
        self.cluster_names = cluster_names
        self.do_plot = do_plot

        super().__init__(file_name, save_class)

    def makePopos(self):
        self.bag_of_clusters = SaveLoadPOPO(self.bag_of_clusters, self.output_folder + self.file_name + "_boc.npy", "npy")
        self.pav_classes = SaveLoadPOPO(self.pav_classes, self.output_folder + self.file_name + "_pav.npy", "npy")

    def plot(self, x, y, y_):
        segments = [[[i, y[i]], [i, y_[i]]] for i in range(len(x))]
        lc = LineCollection(segments, zorder=0)
        lc.set_array(np.ones(len(y)))
        lc.set_linewidths(0.5 * np.ones(len(x)))
        fig = plt.figure()
        plt.plot(x, y, 'r.', markersize=2)
        plt.plot(x, y_, 'g.', markersize=12)
        plt.legend(('Data', 'Isotonic Fit'), loc='lower right')
        plt.title('Isotonic regression')
        plt.show()

    def makePopoArray(self):
        self.popo_array = [self.bag_of_clusters, self.pav_classes]

    def getLROnBag(self, cluster_dict, names, sparse_freqs):
        bag_of_clusters = []
        # Note, prior we used the PPMI values directly here somehow...
        for c in range(len(cluster_dict)):
            # Remove the colons
            for f in range(len(cluster_dict[c])):
                if ":" in cluster_dict[c][f]:
                    cluster_dict[c][f] = cluster_dict[c][f][:-1]
            # Add all of the frequences together to make a bag-of-clusters
            name = cluster_dict[c][0]
            word_array = sparse_freqs[names[name]].toarray()
            accum_freqs = np.zeros(shape=len(word_array), dtype=np.int64)
            # For all the cluster terms
            for name in cluster_dict[c]:
                if ":" in name:
                    name = name[:-1]
                # Import the class
                class_to_add = sparse_freqs[names[name]].toarray()
                # Add the current class to the older one
                accum_freqs = np.add(accum_freqs, class_to_add)
            # Append this clusters frequences to the group of them
            bag_of_clusters.append(accum_freqs)
        # Convert to binary
        for c in range(len(bag_of_clusters)):
            bag_of_clusters[c][bag_of_clusters[c] > 1] = 1
            bag_of_clusters[c] = bag_of_clusters[c][
                0]  # For some reason the internal arrays are the single element of another array
        return bag_of_clusters

    def process(self):
        c_name_array = []
        # Process
        for i in range(len(self.cluster_names)):
            c_name_array.append(self.cluster_names[i].split())
        self.bag_of_clusters.value = self.getLROnBag(c_name_array, self.bow_names, self.sparse_freqs)

        self.pav_classes.value = []


        for f in range(len(self.bag_of_clusters.value)):
            print(self.cluster_names[f])
            x = np.asarray(self.bag_of_clusters.value[f])
            y = self.ranking[f]

            ir = IsotonicRegression()
            y_ = ir.fit_transform(x, y)
            self.pav_classes.value.append(y_)
            if self.do_plot:
                self.plot(x, y, y_)
            print(f)
        self.pav_classes.value = np.asarray(self.pav_classes.value)


    def getPAV(self):
        if self.processed is False:
            self.pav_classes.value = self.save_class.load(self.pav_classes)
        return self.pav_classes.value