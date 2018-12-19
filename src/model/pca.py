
from sklearn.decomposition import TruncatedSVD

def getPCA(tf, dim):
    svd = TruncatedSVD(n_components=dim) # use the scipy algorithm "arpack"
    pos = svd.fit_transform(tf)
    return pos

