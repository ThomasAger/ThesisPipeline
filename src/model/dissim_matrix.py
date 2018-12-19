import numpy as np
import scipy.sparse as sp
from math import pi

# Just a function to import a 2d array. specify the file_type as f = float, i = integer. In this case its a float.
def import2dArray(file_name, file_type="f", return_sparse=False):
    if file_name[-4:] == ".npz":
        print("Loading sparse array")
        array = sp.load_npz(file_name)
        if return_sparse is False:
            array = array.toarray()
    elif file_name[-4:] == ".npy":
        print("Loading numpy array")
        array = np.load(file_name)#
    else:
        with open(file_name, "r") as infile:
            if file_type == "i":
                array = [list(map(int, line.strip().split())) for line in infile]
            elif file_type == "f":
                array = [list(map(float, line.strip().split())) for line in infile]
            elif file_type == "discrete":
                array = [list(line.strip().split()) for line in infile]
                for dv in array:
                    for v in range(len(dv)):
                        dv[v] = int(dv[v][:-1])
            else:
                array = np.asarray([list(line.strip().split()) for line in infile])
        array = np.asarray(array)
    print("successful import", file_name)
    return array

# Just a function to write a 2d array to a text file. Also produces a numpy file
def write2dArray(array, name):
    try:
        file = open(name, "w")
        print("starting array")
        for i in range(len(array)):
            for n in range(len(array[i])):
                file.write(str(array[i][n]) + " ")
            file.write("\n")
        file.close()
    except FileNotFoundError:
        print("FAILURE")
    try:
        if name[-4:] == ".txt":
            name = name[:-4]
        array = np.asarray(array)
        np.save(name, array)
    except:
        print("failed")

    print("successful write", name)

def calcAng(e1, e2, norm1, norm2):
    return (2 / pi) * np.arccos(np.dot(e1, e2) / (norm1 * norm2))
import math

def getDsimMatrixDense(tf):
    #tf = np.asarray(tf.astype(np.float32).transpose().toarray())
    tf = np.asarray(tf.todense(), dtype=np.float32)
    docs_len = tf.shape[0]
    if tf.shape[0] > tf.shape[1]:
        print(tf.shape, "DOCS:", docs_len)
        raise ValueError("Probably wrong")
    dm = np.zeros([docs_len, docs_len], dtype=np.float32)
    dm2 = np.zeros([docs_len, docs_len], dtype=np.float32)
    norms = np.zeros(docs_len, dtype=np.float32)
    # Calculate norms
    for ei in range(docs_len):
        norms[ei] = np.linalg.norm(tf[ei])
        if ei % 1000 == 0:
            print("norms", ei)
    for i in range(docs_len):
        for j in range(i+1):

            dm[i][j] = calcAng(tf[i], tf[j], norms[i], norms[j])
            if math.isnan(dm[i][j]):
                dm[i][j] = 0.0
            #if j %1000 == 0:
            #    print("j", j)
        print("i", i)

    # Fill in the values of the mirrored array
    cr = 0
    for c in range(docs_len):
        for r in range(cr+1, docs_len):
            if math.isnan(dm[cr][r]):
                dm[cr][r] = 0
            dm[cr][r] = dm[r][c]
        cr += 1

    return dm

if __name__ == '__main__':
    term_frequency_fn = "bow path" # This is your bag of words
    dm_fn = "output path" # This is where you want the space to output. Make sure it exists
    tf = import2dArray(term_frequency_fn)
    dm = getDsimMatrixDense(tf)
    write2dArray(dm, dm_fn)