import os
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy import sparse as sp

def write2dCSV(array, name):
    file = open(name, "w")

    for i in range(len(array[0])):
        if i >= len(array[0]) - 1:
            file.write(str(i) + "\n")
        else:
            file.write(str(i) + ",")
    for i in range(len(array)):
        for n in range(len(array[i])):
            if n >= len(array[i])-1:
                file.write(str(array[i][n]))
            else:
                file.write(str(array[i][n]) + ",")
        file.write("\n")
    file.close()



def writeCSV(features, classes, class_names, file_name, header=True):
    for c in range(len(class_names)):
        file = open(file_name + class_names[c] + ".csv", "w")
        if header:
            for i in range(len(features[0])):
                if i >= len(features[0]) - 1:
                    file.write(str(i) + "," + class_names[c] + "\n")
                else:
                    file.write(str(i) + ",")
        for i in range(len(features)):
            for n in range(len(features[i])):
                if n >= len(features[i]) - 1:
                    if classes[c][i] == 0:
                        file.write(str(features[i][n]) + ",FALSE")
                    else:
                        file.write(str(features[i][n]) + ",TRUE")
                else:
                    file.write(str(features[i][n]) + ",")
            file.write("\n")
        file.close()


def writeArff(features, classes, class_names, file_name, header=True):
    for c in range(len(class_names)):
        file = open(file_name + class_names[c] + ".arff", "w")
        file.write("@RELATION genres\n")
        if header:
            for i in range(len(features[0])):
                file.write("@ATTRIBUTE " + str(i) + " NUMERIC\n")
            file.write("@ATTRIBUTE " + class_names[c] + " {f,t}\n")
        file.write("@DATA\n")
        for i in range(len(features)):
            for n in range(len(features[i])):
                if n >= len(features[i]) - 1:
                    if classes[c][i] == 0:
                        file.write(str(features[i][n]) + ",f")
                    else:
                        file.write(str(features[i][n]) + ",t")
                else:
                    file.write(str(features[i][n]) + ",")
            file.write("\n")
        file.close()


def get_CSV_from_arrays(arrays, col_names):
    if len(arrays) != len(col_names):
        raise ValueError("Inconsistent lengths", len(arrays), len(col_names))
    dict = OrderedDict()
    for i in range(len(col_names)):
        dict[col_names[i]] = arrays[i]
    csv_df = pd.DataFrame(dict)
    return csv_df.to_csv()

def write_string(string, name):
    try:
        file = open(name, "w")
        file.write(string)
        file.close()
    except FileNotFoundError:
        print("Failed")

def write1dArray(array, name, encoding=None):
    file = open(name, "w", encoding=encoding)
    for i in range(len(array)):
        file.write(str(array[i]) + "\n")
    file.close()

def write(value, name, encoding=None):
    file = open(name, "w", encoding=encoding)
    file.write(str(value) + "\n")
    file.close()

def write1dLinux(array, name):
    file = io.open(name, "w", newline='\n')
    for i in range(len(array)):
        file.write(str(array[i]) + "\n")
    file.close()


def write1dCSV(array, name):
    file = open(name, "w")
    file.write("0\n")
    for i in range(len(array)):
        file.write(str(array[i]) + "\n")
    file.close()

def write_to_csv_key(csv_fn, col_names, cols_to_add, keys):
    df = pd.read_csv(csv_fn, index_col=0)
    for c in range(len(cols_to_add)):
        for k in range(len(keys)):
            df[col_names[c]][k] = cols_to_add[c][k]
    df.to_csv(csv_fn)

def write_csv(csv_fn, col_names, cols_to_add, key):
    d = {}
    for c in range(len(cols_to_add)):
        print(c, "/", len(cols_to_add))
        d[col_names[c]] = cols_to_add[c]
    df = pd.DataFrame(d, index=key)
    df.to_csv(csv_fn)

def read_csv(csv_fn):
    csv = pd.read_csv(csv_fn, index_col=0)
    return csv

def csv_pd_to_array(csv_pd):
    print("")
    return [csv_pd._info_axis.values, csv_pd.values[0], csv_pd._stat_axis.values]
"""
write_to_csv("../../data/newsgroups/rules/tree_csv/sns_ppmi3mdsnew200svmdualCV1S0 SFT0 allL03018836 LR acc KMeans CA400 MC1 MS0.4 ATS500 DS800 newsgroupsAVG.csv", "1", "1")
for i in range(2147000000):
    print(i)
"""


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


def fnsExist(all_fns):
    all_exist = 0
    for f in range(len(all_fns)):
        if os.path.exists(all_fns[f]):
            print(all_fns[f], "Already exists")
            all_exist += 1
        else:
            print(all_fns[f], "Doesn't exist")
    if all_exist == len(all_fns):
        return True
    return False


def toBool(string):
    if string == "True":
        return True
    else:
        return False




def save_dict(dct, file_name):
    file = open(file_name, "w")
    for key, value in dct.items():
        try:
            file.write(str(key) + " " + str(value) + " " + str(type(value)) + "\n")
        except UnicodeEncodeError:
            print("Unicode error")
        print(str(key) + " " + str(value) + " " + str(type(value)) + "\n")
    file.close()


def load_dict(file_name):
    dict = {}
    with open(file_name, "r", encoding="cp1252") as infile:
        lines = infile.readlines()
        for l in lines:
            split = l.split()
            new_type = False
            if split[1] == "None":
                dict[split[0]] = None
            try:
                type = split[3]
                new_type = True
                if 'float' in type:
                    dict[split[0]] = float(split[1])

                if "str" in type:
                    dict[split[0]] = str(split[1])

                if "int" in type:
                    dict[split[0]] = int(split[1])

                if "None" in type:
                    dict[split[0]] = None

                if "bool" in type:
                    dict[split[0]] = bool(split[1])
            except IndexError:
                print("No type in dict")
            if not new_type:
                dict[split[0]] = split[1]
    return dict

def loadNpyDict(fn):
    return np.load(fn).item()

def writeArrayDict(dict, name):
    file = open(name, "w")
    for key, value in dict.itsems():
        file.write(str(key) + ": ")
        for v in value:
            file.write(str(v) + " ")
        file.write("\n")
    file.close()


def writeArrayDict1D(dict, name):
    file = open(name, "w")
    for key, value in dict.items():
        file.write(str(key) + ": ")
        file.write(str(value) + " ")
        file.write("\n")
    file.close()


def readArrayDict(file_name):
    file = open(file_name)
    lines = file.readlines()
    dict = OrderedDict()
    for l in lines:
        l = l.split()
        if l[0][len(l[0])-1:] == ":":
            name = l[0][:-1]
        else:
            name = l[0]
        del l[0]
        dict[name] = l
        print(name)
    return dict


def readArrayDict1D(file_name):
    file = open(file_name)
    lines = file.readlines()
    dict = OrderedDict()
    for l in lines:
        l = l.split()
        if l[0][len(l[0])-1:] == ":":
            name = l[0][:-1]
        else:
            name = l[0]
        del l[0]
        dict[name] = int(l)
        print(name)
    return dict





def importFirst2dArray(file_name, file_type="f", amount=100):
    array = []
    with open(file_name, "r") as infile:
        counter = 0
        for line in infile:
            if file_type == "i":
                array.append(list(map(int, line.strip().split())))
            elif file_type == "f":
                array.append(list(map(float, line.strip().split())))
            elif file_type == "discrete":
                to_add = list(line.strip().split())
                for v in range(len(to_add)):
                    to_add[v] = int(to_add[v][:-1])
            else:
                array.append(list(line.strip().split()))
            if counter > amount:
                return array
            counter += 1
    return array


def importTabArray(file_name):
    with open(file_name, "r") as infile:
        string_array = [line.split("\t")[:-1] for line in infile]
    return string_array


def writeTabArray(array, file_name):
    names_with_tabs = []
    for name_array in array:
        string_to_append = ""
        for n in name_array:
            string_to_append = string_to_append + n + "\t"
        names_with_tabs.append(string_to_append)
    write1dArray(names_with_tabs, file_name)


def getFns(folder_path):
    file_names = []
    onlyfiles = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    for i in onlyfiles:
        file_names.append(i)
    return file_names


def importArray(file_name, file_type="s", return_sparse=True):
    if file_name[-4:] == ".npz":
        if return_sparse is False:
            return sp.load_npz(file_name).toarray()
        return sp.load_npz(file_name)
    elif file_name[-4:] == ".npy":
        return np.load(file_name)#
    else:
        with open(file_name, "r") as infile:
            lines = infile.readlines()
            split = True
            for line in lines:
                val = list(line.split())
                if len(val) == 1:
                    split = False
            if split is True:
                if file_type == "i":
                    return [list(map(int, line.strip().split())) for line in lines]
                elif file_type == "f":
                    return [list(map(float, line.strip().split())) for line in lines]
                else:
                    return np.asarray([list(line.strip().split()) for line in lines])
            else:
                if file_type == "i":
                    return [int(line.strip()) for line in lines]
                elif file_type == "f":
                    return [float(line.strip()) for line in lines]
                else:
                    return [line.strip() for line in lines]
def isArray(N):
    if hasattr(N, '__len__') and (not isinstance(N, str)):
        return True
    else:
        return False

def writeArray(array, name):
    if isArray(array[0]):
        write2dArray(array, name)
    else:
        write1dArray(array, name)
    print("successful write", name)


def getFolder(folder_path):
    onlyfiles = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    two_d = []
    for name in onlyfiles:
        one_d = import1dArray(folder_path + name)
        two_d.append(one_d)
    return two_d


"""
def importLargeTextFile(file_name, file_type="s"):
    if file_type == "s":
        # Setting to something other than np.object results in a memory error. Could be because of something weird?
        array = np.loadtxt(file_name, dtype=np.object, delimiter="\n")
        # This method adds 'b\\' to the start of each string
        # To show that they are bytes. However, methods that don't do this
        # Attempt to convert it into an array. So, this ends up being
        # The most memory efficient way to get this into a manageable numpy string
        # Simply because its loaded directly into a numpy array.

        if 'b\\' in array[0][:1]:
            for i in range(len(array)):
                array[i] = array[i][3:]
    else:
        with open(file_name, "r", encoding="cp1252") as infile:
            print("opened as file")
            if file_type == "f":
                array = []
                lines = infile.readlines()
                for line in lines:
                    array.append(float(line.strip()))
            elif file_type == "i":
                array = [int(float(line.strip())) for line in infile]
        np.asarray(array)
    return array
"""
def import1dArray(file_name, file_type="s"):
    with open(file_name, "r", encoding="cp1252") as infile:
        if file_type == "f":
            array = []
            lines = infile.readlines()
            for line in lines:
                array.append(float(line.strip()))
        elif file_type == "i":
            array = [int(float(line.strip())) for line in infile]
        else:
            array = [line.strip() for line in infile]
    return np.asarray(array)

def importValue(file_name, file_type="s"):
    with open(file_name, "r", encoding="cp1252") as infile:
        if file_type == "f":
            lines = infile.readlines()
            for line in lines:
                value = float(line.strip())
        elif file_type == "i":
            lines = infile.readlines()
            for line in lines:
                value = int(line.strip())
        else:
            lines = infile.readlines()
            for line in lines:
                value = line.strip()
    return value

def import2dArray(file_name, file_type="f", return_sparse=False):
    if file_name[-4:] == ".npz":
        print("Loading sparse array")
        array = sp.load_npz(file_name)
        if return_sparse is False:
            array = array.toarray()
    elif file_name[-4:] == ".npy":
        print("Loading numpy array")
        array = np.load(file_name)
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

def importNumpyVectors(numpy_vector_path=None):
    movie_vectors = np.load(numpy_vector_path)
    movie_vectors = np.ndarray.tolist(movie_vectors)
    movie_vectors = list(reversed(zip(*movie_vectors)))
    movie_vectors = np.asarray(movie_vectors)
    return movie_vectors




def save_averages_and_final_csv(dict_array, class_names, average_file_names, output_folder, end_file_name):
    for i in range(int(len(dict_array))):
        save_csv_from_dict(dict_array[i], class_names, output_folder + average_file_names[i] + ".csv")
    save_averages_from_dicts(dict_array, average_file_names, end_file_name)


def save_averages_from_dicts(score_dicts, row_names, file_name):
    rows = []
    col_names = []
    for sd in range(len(score_dicts)):
        csv_acc = None
        csv_aurocs = None
        csv_recalls = None
        csv_precs = None
        csv_kappas = None
        csv_f1s = None

        if "avg_acc" in score_dicts[sd]:
            csv_acc = score_dicts[sd]["avg_acc"]
        if "avg_f1" in score_dicts[sd]:
            csv_f1s = score_dicts[sd]["avg_f1"]
        if "avg_auroc" in score_dicts[sd]:
            csv_aurocs = score_dicts[sd]["avg_auroc"]
        if "avg_recall" in score_dicts[sd]:
            csv_recalls = score_dicts[sd]["avg_recall"]
        if "avg_prec" in score_dicts[sd]:
            csv_precs = score_dicts[sd]["avg_prec"]
        if "avg_kappa" in score_dicts[sd]:
            csv_kappas = score_dicts[sd]["avg_kappa"]

        col_data = []

        if csv_f1s is not None:
            col_data.append(csv_f1s)
        if csv_acc is not None:
            col_data.append(csv_acc)
        if csv_aurocs is not None:
            col_data.append(csv_aurocs)
        if csv_precs is not None:
            col_data.append(csv_precs)
        if csv_kappas is not None:
            col_data.append(csv_kappas)
        if csv_recalls is not None:
            col_data.append(csv_recalls)

        rows.append(col_data)

        if sd == 0:
            if csv_f1s is not None:
                col_names.append("avg_f1")
            if csv_acc is not None:
                col_names.append("avg_acc")
            if csv_aurocs is not None:
                col_names.append("avg_auroc")
            if csv_precs is not None:
                col_names.append("avg_prec")
            if csv_kappas is not None:
                col_names.append("avg_kappa")
            if csv_recalls is not None:
                col_names.append("avg_recall")
    rows = np.asarray(rows).transpose()
    write_csv(file_name, col_names, rows, row_names)
def save_csv_from_dict(score_dict, class_names, csv_fn):

    csv_acc = None
    csv_aurocs = None
    csv_recalls = None
    csv_precs = None
    csv_kappas = None
    csv_f1s = None

    if "acc" in score_dict:
        csv_acc = score_dict["acc"]
        csv_acc = np.append(csv_acc, score_dict["avg_acc"])
    if "f1" in score_dict:
        csv_f1s = score_dict["f1"]
        csv_f1s = np.append(csv_f1s, score_dict["avg_f1"])
    if "auroc" in score_dict:
        csv_aurocs = score_dict["auroc"]
        csv_aurocs = np.append(csv_aurocs, score_dict["avg_auroc"])
    if "recall" in score_dict:
        csv_recalls = score_dict["recall"]
        csv_recalls = np.append(csv_recalls, score_dict["avg_recall"])
    if "prec" in score_dict:
        csv_precs = score_dict["prec"]
        csv_precs = np.append(csv_precs, score_dict["avg_prec"])
    if "kappa" in score_dict:
        csv_kappas = score_dict["kappa"]
        csv_kappas = np.append(csv_kappas, score_dict["avg_kappa"])

    col_data = []
    col_names = []

    if csv_f1s is not None:
        col_data.append(csv_f1s)
        col_names.append("f1")
    if csv_acc is not None:
        col_data.append(csv_acc)
        col_names.append("acc")
    if csv_aurocs is not None:
        col_data.append(csv_aurocs)
        col_names.append("auroc")
    if csv_precs is not None:
        col_data.append(csv_precs)
        col_names.append("prec")
    if csv_kappas is not None:
        col_data.append(csv_kappas)
        col_names.append("kappa")
    if csv_recalls is not None:
        col_data.append(csv_recalls)
        col_names.append("recall")

    class_names = np.append(class_names, "AVERAGE")
    write_csv(csv_fn, col_names, col_data, class_names)


