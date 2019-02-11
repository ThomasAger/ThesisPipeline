from util import io as dt
import cProfile
import numpy as np
import numbers
from util import py
import scipy.sparse as sp
from common.Method import Method
from common.SaveLoadPOPO import SaveLoadPOPO
# Data structure management tasks



def transIfRowsLarger(mat):
    if mat is not None:
        if py.isArray(mat[0]) is True:
            print("Matrix in format", len(mat), len(mat[0]))
            if len(mat) > len(mat[0]):
                print("Rows larger than columns, transposing")
                mat = mat.transpose()
                print("Matrix now in format", len(mat), len(mat[0]))
                if len(mat) > len(mat[0]):
                    raise ValueError("Matrix transposed but did not change shape, error in input (One/some arrays in mat are probably larger or smaller than the others)")
    return mat

def transIfColsLarger(mat):
    if mat is not None:
        if py.isArray(mat[0]) is True:
            print("Matrix in format", len(mat), len(mat[0]))
            if len(mat) < len(mat[0]):
                print("Columns larger than rows, transposing")
                mat = mat.transpose()
                print("Matrix now in format", len(mat), len(mat[0]))
                if len(mat) < len(mat[0]):
                    raise ValueError("Matrix transposed but did not change shape, error in input (One/some arrays in mat are probably larger or smaller than the others)")
    return mat

def parameter_list_to_dict_str(parameter_list_string):#
    dict_str = ["param_dict = {"]
    for i in range(len(parameter_list_string)):
        str = ""
        if parameter_list_string[i][:1] == "#":
            continue
        else:
            split = parameter_list_string[i].split()
            if len(split) == 0:
                continue
            str += "\t'" + split[0] + "': " + split[0] + ","
        dict_str.append(str)
    dict_str.append("}")
    return dict_str

def isFloat(x):
    if isinstance(x, float):
        return True
    return False
def isInt(x):
    if isinstance(x, numbers.Integral):
        return True
    return False

def isStr(x):
    if isinstance(x, str):
        return True
    return False

# If it's a list and not a string
def isList(x):
    try:
        len(x)
    except TypeError:
        return False
    try:
        x.capitalize()
    except AttributeError:
        return True
    return False

def isArray(N):
    if hasattr(N, '__len__') and (not isinstance(N, str)):
        return True
    return False


def shorten2dFloats(floats_fn):
    fa = dt.import2dArray(floats_fn)
    for a in range(len(fa)):
        fa[a] = np.around(fa[a], decimals=4)
    return fa

def shortenFloatsNoFn(fa):
    for a in range(len(fa)):
        fa[a] = np.around(fa[a], decimals=4)
    return fa
def deleteAllButIndexes(array, indexes):
    old_ind = list(range(len(array)))
    del_ind = np.delete(old_ind, indexes)
    array = np.delete(array, del_ind)
    return array


def remove_indexes(indexes, array_fn):
    array = np.asarray(dt.import1dArray(array_fn))
    array = np.delete(array, indexes, axis=0)
    dt.write1dArray(array, array_fn)
    print("wrote", array_fn)

import re


def concatenateArrays(arrays, file_name):
    new_array = arrays[0]
    for a in range(1, len(arrays)):
        new_array = np.concatenate((new_array, arrays[a]), axis=0)
    dt.write2dArray(new_array, file_name)


def getDifference(array1, array2):
    file1 = open(array1)
    file2 = open(array2)
    for line1 in file1:
        line1 = line1.split()
        line1 = [str(line1[v]) for v in range(len(line1))]
        print(line1)
        for line2 in file2:
            line2 = line2.split()
            line2 = [str(line2[v]) for v in range(len(line2))]
            print(line2)
            break
        break

def getDifference(array1, array2):
    file2 = dt.import1dArray(array1)
    file1 = dt.import1dArray(array2)
    for line1 in file1:
        found = False
        for line2 in file2:
            if line2 == line1:
                found = True
                break
        if not found:
            print(line1)


def mean_of_array(array):
    total = []
    for a in array[0]:
        total.append(a)
    len_array = len(array)
    for a in range(1, len_array):
        for v in range(0, len(array[a])):
            total[v] = total[v] + array[a][v]
    for v in range(len(total)):
        divided = (total[v] / len_array)
        total[v] = divided
    return total


def checkIfInArray(array, thing):
    for t in array:
        if thing == t:
            return True
    return False

def getIndexInArray(array, thing):
    for t in range(len(array)):
        if thing == array[t]:
            return t
    return None

def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

def sortByArray(array_to_sort, array_to_sort_by):
    Y = array_to_sort_by
    X = array_to_sort
    sorted_array = [x for (y, x) in sorted(zip(Y, X))]
    return sorted_array

def sortByReverseArray(array_to_sort, array_to_sort_by):
    Y = array_to_sort_by
    X = array_to_sort
    sorted_array = [x for (y, x) in reversed(sorted(zip(Y, X)))]
    return sorted_array


def convertToFloat(string_array):
    temp_floats = []
    for string in string_array:
        float_strings = string.split()
        i = 0
        for float_string in float_strings:
            float_strings[i] = float(float_string)
            i = i + 1
        temp_floats.append(float_strings)
    return temp_floats


def convertLine(line):
    line = list(map(float, line.strip().split()))
    return line


def findDifference(string1, string2):
    index = 0
    for l in range(len(string1)):
        if string1[l] != string2[l]:
            index = l
            break
        else:
            index = len(string1)
    if len(string1[index:]) < 5:
        print(string1[index:])
    if len(string2[index:]) < 5:
        print(string2[index:])

def reverseArrays(md_array):
    md_array = np.asarray(md_array)
    reversed_array = []
    for a in md_array:
        reversed_array.append(a[::-1])
    return reversed_array



if __name__ == '__main__':
    """
    parameter_list_string = io.import1dArray("../../data/parameter_list_string.txt")
    parameter_dict = parameter_list_to_dict_str(parameter_list_string)
    io.write1dArray(parameter_dict, "../../data/parameter_dict.txt")
    print(isInt(0.0), isInt(1.0), isInt(1.5), isInt(-1), isInt(None))
    print(isFloat(0.0), isFloat(1.0), isFloat(1.5), isFloat(-1), isFloat(None))
    print("------------------COO")
    cProfile.run('sp.rand(m=50000, n=40000, format="coo", dtype=np.float32)')
    print("------------------CSR")
    cProfile.run('sp.rand(m=50000, n=40000, format="csr", dtype=np.float32)')
    print("------------------COO converted to CSR")
    cProfile.run('sp.csr_matrix(sp.rand(m=50000, n=40000, format="coo", dtype=np.float32))')
    """
    orig_fn = "../../data/processed/reuters/rep/mds/"
    dim = 200
    t_a = dt.import2dArray(orig_fn + "num_stw_"+str(dim)+"_MDS.txt")
    np.save(orig_fn + "num_stw_"+str(dim)+"_MDS.npy", t_a)
    dim = 100
    t_a = dt.import2dArray(orig_fn + "num_stw_"+str(dim)+"_MDS.txt")
    np.save(orig_fn + "num_stw_"+str(dim)+"_MDS.npy", t_a)
    dim = 50
    t_a = dt.import2dArray(orig_fn + "num_stw_"+str(dim)+"_MDS.txt")
    np.save(orig_fn + "num_stw_"+str(dim)+"_MDS.npy", t_a)

    #print(isList([1,2,3]), isList(np.zeros(shape=(3,4))), isList("fartr"), isList(5), isList(sparse_m))