import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from sklearn.decomposition import TruncatedSVD

import io.io


def makeTopVectors(filename):

    vectors = io.io.import2dArray("Rankings/" + filename + ".representation")
    top250names = io.io.import1dArray("filmdata/top250.txt")
    film_names = io.io.import1dArray("filmdata/filmNames.txt")


    indexes = []
    ordered_names = []
    for f in range(len(film_names)):
        for t in top250names:
            if film_names[f] == t:
                indexes.append(f)
                ordered_names.append(t)

    top_vectors = [[]]
    for v in range(len(vectors)):
        if v > 0:
            top_vectors.append([])
        for i in range(len(vectors[v])):
            for id in indexes:
                if i == id:
                    top_vectors[v].append(vectors[v][i])

    io.io.write2dArray(top_vectors, "Plots/Top174" + filename + ".representation")
    io.io.write1dArray(ordered_names, "Plots/Top174OrderedByOriginalList.txt")

def makeTopVectorsDirections(filename):
    vectors = io.io.import2dArray("Directions/" + filename + "Cut.directions")
    top250names = io.io.import1dArray("filmdata/top250.txt")
    filmnames = io.io.import1dArray("filmdata/filmNames.txt")

    top250vectors = []

    for f in range(len(filmnames)):
        for t in range(len(top250names)):
            if filmnames[f] == top250names[t]:
                top250vectors.append(vectors[t])

    io.io.write2dArray(top250vectors, "../data/movies/plot/t250" + filename + ".directions")


def plotTopVectors(filename):

    names = io.io.import1dArray("../data/movies/plot/Top174OrderedByOriginalList.txt")
    space = io.io.import2dArray("../data/movies/plot/Top174" + filename + ".representation")

    svd = TruncatedSVD(n_components=2, random_state=42)

    svd_space = svd.fit_transform(space)
    pl.plot(space[0], 'rx')
    pl.show()


    """
    pca = PCA(n_components=2)
    pca.fit_transform(representation)
    print representation
    for s in representation:
        print s
    print pca.explained_variance_ratio_
    for s in pca.explained_variance_ratio_:
        print s
    """


def plotClusters(filename):
    names = io.io.import1dArray("Plots/Top174OrderedByOriginalList.txt")
    space = io.io.import2dArray("Plots/Top174" + filename + ".representation")
    cluster_names = io.io.import1dArray("Clusters/films100N0.6H25L3CutLeastSimilarHIGH0.75,0.67.names")

    #svd = TruncatedSVD(n_components=2, random_state=42)

    cx = 8
    cy = 9
    x = []
    y = []
    for s in space[cx]:
        x.append(s)
    for s in space[cy]:
        y.append(s)

    #svd_space = svd.fit_transform(representation)

    fig, ax = plt.subplots()
    ax.scatter(x, y, picker=True)
    #for i, name in enumerate(found_names):
    #    ax.annotate(name, (x[i], y[i]))
    ax.set_xlabel(cluster_names[cx])
    ax.set_ylabel(cluster_names[cy])
    def onpick3(event):
        ind = event.ind
        print('onpick3 scatter:', names[ind[0]])

    fig.canvas.mpl_connect('pick_event', onpick3)

    plt.show()

def plotSVD(filename):
    names = io.io.import1dArray("Plots/Top174OrderedByOriginalList.txt")
    space = io.io.import2dArray("Plots/Top174" + filename + ".representation")

    space = np.matrix.transpose(np.asarray(space))
    space = space.tolist()
    svd = TruncatedSVD(n_components=2, random_state=42)
    svd_space = svd.fit_transform(space)

    x = []
    y = []

    for s in svd_space:
        print(s)
        x.append(s[0])
        y.append(s[1])

    fig, ax = plt.subplots()
    ax.scatter(x, y, picker=True)
    # for i, name in enumerate(found_names):
    #    ax.annotate(name, (x[i], y[i]))

    def onpick3(event):
        ind = event.ind
        print('onpick3 scatter:', names[ind[0]])

    fig.canvas.mpl_connect('pick_event', onpick3)

    plt.show()

plotSVD("films100")
plotSVD("films200L1100N0.5")
plotSVD("films200L1100N0.5TermFrequencyN0.5FT")
# Get the PCA of top 200 films

# Plot the PCA values on a graph

# If the films are too close together, delete the least popular one

#dist = np.linalg.norm(a-b)



