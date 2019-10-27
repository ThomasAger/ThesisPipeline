import os
import unicodedata

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from util.io import write1dArray, read_csv, write_csv, write2dArray, import2dArray, getFns, import1dArray

"""

DATA IMPORTING TASKS

"""


###### GLOBAL DATA UTILITIES #######




####### MANAGING PROJECT-SPECIFIC TASKS ######

def balanceClasses(movie_vectors, class_array):
    count = 0
    count2 = 0
    for i in class_array:
        if i == 0:
            count+=1
        else:
            count2+=1
    indexes_to_remove = []
    amount_to_balance_to = count-count2
    amount = 0
    while amount < amount_to_balance_to:
        index = random.randint(0, len(class_array) - 1)
        if class_array[index] == 0:
            indexes_to_remove.append(index)
            amount+=1
    movie_vectors = np.delete(movie_vectors, indexes_to_remove, axis=0)
    class_array = np.delete(class_array, indexes_to_remove)

    return movie_vectors, class_array


def balanceClasses(movie_vectors, class_array):
    count = 0
    count2 = 0
    for i in class_array:
        if i == 0:
            count+=1
        else:
            count2+=1
    indexes_to_remove = []
    amount_to_balance_to = count - count2*2
    amount = 0
    while amount < amount_to_balance_to:
        index = random.randint(0, len(class_array) - 1)
        if class_array[index] == 0:
            indexes_to_remove.append(index)
            amount+=1
    movie_vectors = np.delete(movie_vectors, indexes_to_remove, axis=0)
    class_array = np.delete(class_array, indexes_to_remove)

    return movie_vectors, class_array

def balance2dClasses(movie_vectors, movie_classes, min_occ):

    indexes_to_remove = []
    for m in range(len(movie_classes)):
        counter = 0
        for i in movie_classes[m]:
            if i > 0:
                counter+=1
        if counter < min_occ:
            indexes_to_remove.append(m)

    movie_vectors = np.delete(movie_vectors, indexes_to_remove, axis=0)
    movie_classes = np.delete(movie_classes, indexes_to_remove, axis=0)
    print("deleted", len(indexes_to_remove))
    return movie_vectors, movie_classes


"""

DATA EDITING TASKS

"""





def splitData(training_data, movie_vectors, movie_labels):
    x_train = np.asarray(movie_vectors[:training_data])
    y_train = np.asarray(movie_labels[:training_data])
    x_test = np.asarray(movie_vectors[training_data:])
    y_test = np.asarray(movie_labels[training_data:])
    return  x_train, y_train,  x_test, y_test





"""
a = import2dArray("D:\Eclipse\MDS/class-all-30-18836-alldm", "f")

a = np.nan_to_num(a)

write2dArray(a, "class-all-30-18836-alldmnTn")
"""
"""
mds = import2dArray("../data/newsgroups/nnet/spaces/mds.txt")

mds = mds.transpose()

write2dArray(mds, "../data/newsgroups/nnet/spaces/mds.txt")
"""
def sortIndexesByArraySize(array):
    array_of_lens = []
    for i in range(len(array)):
        array_of_lens.append(len(array[i]))
    indexes = [i[0] for i in sorted(enumerate(array_of_lens), key=lambda x: x[1])]
    return indexes


"""
csv_fns = []
for i in range(5):
    csv_fns.append("../data/wines/rules/tree_csv/" +
        "wines ppmi E200 DS[100, 100, 100] DN0.5 HAtanh CV5 S0 SFT0L050 ndcg0.9001100" + ".csv")
        """
#average_csv(csv_fns)

"""
file_path = "../data/movies/LDA/Names/"
file_names = getFns(file_path)
for fn in file_names:
    md_array = import2dArray(file_path + fn, "s")
    reversed_array = reverseArrays(md_array)
    write2dArray(reversed_array, file_path + fn)
"""

def removeIndexes(file_name, indexes, type="f"):
    removed_indexes = []
    orig_array = import2dArray(file_name, type)
    removed_indexes = np.delete(orig_array, indexes, axis=0)
    write2dArray(removed_indexes, file_name[:-4] + "removedind.txt")
"""
indexes = [121,
144,
64,
60,
58,
45,
42,
41,
40,
38,
37,
35,
33,
32,
15,
14,
12,
10,
7,
5,
2]

for i in range(len(indexes)):
    indexes[i] = indexes[i] -1

removeIndexes("../data/newsgroups/cluster/dict/n100mdsnnetE400DS[100]DN0.5CTnewsgroupsHAtanhCV1 S0OA softmax SFT0 allL030ndcg KMeans CA200 MC1 MS0.4 ATS2000 DS400.txt",
              indexes, "s")

removeIndexes("../data/newsgroups/cluster/clusters/n100mdsnnetE400DS[100]DN0.5CTnewsgroupsHAtanhCV1 S0OA softmax SFT0 allL030ndcg KMeans CA200 MC1 MS0.4 ATS2000 DS400.txt",
              indexes)

removeIndexes("../data/newsgroups/cluster/first_term_clusters/n100mdsnnetE400DS[100]DN0.5CTnewsgroupsHAtanhCV1 S0OA softmax SFT0 allL030ndcg KMeans CA200 MC1 MS0.4 ATS2000 DS400.txt",
              indexes)

removeIndexes("../data/newsgroups/cluster/first_terms/n100mdsnnetE400DS[100]DN0.5CTnewsgroupsHAtanhCV1 S0OA softmax SFT0 allL030ndcg KMeans CA200 MC1 MS0.4 ATS2000 DS400.txt",
              indexes, "s")
"""



def getSampledData(property_names, classes, lowest_count, largest_count):
    for yt in range(len(classes)):
        y1 = 0
        y0 = 0
        for y in range(len(classes[yt])):
            if classes[yt][y] >= 1:
                y1 += 1
            if classes[yt][y] == 0:
                y0 += 1

        if y1 < lowest_count or y1 > largest_count:
            classes[yt] = None
            property_names[yt] = None
            print("Deleted", property_names[yt])
            continue

    property_names = [x for x in property_names if x is not None]
    classes = [x for x in classes if x is not None]
    return property_names, classes

def writeClassAll(class_fn, full_phrases_fn, phrases_used_fn, file_name):
    full_phrases = import1dArray(full_phrases_fn)
    #ppmi = np.asarray(import2dArray(class_fn)).transpose()
    ppmi = import2dArray(class_fn)
    new_ppmi = []
    phrases_used = import1dArray(phrases_used_fn)
    for p in range(len(full_phrases)):
        for pi in range(len(phrases_used)):
            if full_phrases[p] == phrases_used[pi]:
                new_ppmi.append(ppmi[p])
                break
    write2dArray(new_ppmi, file_name)
"""
writeClassAll("../data/movies/bow/ppmi/class-all", "../data/movies/bow/phrase_names.txt",
              "../data/movies/bow/names/200.txt", "../data/movies/bow/ppmi/class-all-200")
"""
#writeClassAll("../data/movies/bow/frequency/phrases/class-all", "../data/movies/bow/phrase_names.txt", "../data/movies/svm/names/films100N0.6H75L1200.txt", "../data/movies/bow/frequency/phrases/class-all-200")

"""
sortAndOutput("filmdata/KeywordData/most_common_keywords.txt", "filmdata/KeywordData/most_common_keywords_values.txt",
              "filmdata/KeywordData/most_common_keywordsSORTED.txt", "filmdata/KeywordData/most_common_keyword_valuesSORTED.txt")
"""

"""
top250 = []
for s in import1dArray("filmdata/Top 250 movies.txt"):
    s = s.split()[3:]
    s[len(s)-1] = s[len(s)-1][1:-1]
    s = " ".join(s)
    top250.append(s)
write1dArray(top250, "filmdata/top250.txt")
"""

#write1dArray(getFns("../data/movies/bow/binary/phrases/"), "../data/movies/bow/phrase_names.txt")
# Finding the differences between two entities with different bag of words, but similar terms
def getIndexOfCommonElements(short_list, long_list):
    index = []
    for n in range(len(short_list)):
        for ni in range(len(long_list)):
            if short_list[n] == long_list[ni]:
                index.append(ni)
    return index


def getScoreDifferences(name_word_file1, name_score_file1, name_word_file2, name_score_file2, name, data_type):

    scores1 = import1dArray(name_score_file1, "f")
    scores2 = import1dArray(name_score_file2, "f")

    words1 = import1dArray(name_word_file1, "s")
    words2 = import1dArray(name_word_file2, "s")

    differences_list = []
    if len(words1) > len(words2):
        same_element_index = getIndexOfCommonElements(words2, words1)
        scores1 = np.asarray(scores1)[same_element_index]
        words1 = np.asarray(words1)[same_element_index]
    else:
        same_element_index = getIndexOfCommonElements(words1, words2)
        scores2 = np.asarray(scores2)[same_element_index]
        words2 = np.asarray(words2)[same_element_index]

    for i in range(len(scores1)):
        differences_list.append(scores1[i] - scores2[i])
    most_different_words = [x for (y,x) in sorted(zip(differences_list,words1))]
    differences_list = sorted(differences_list)
    write1dArray(most_different_words, "../data/" + data_type + "/SVM/difference/most_different_words_" + name + ".txt")
    write1dArray(differences_list, "../data/" + data_type + "/SVM/difference/most_different_values_" + name + ".txt")
data_type = "placetypes"
filepath = "../data/"+data_type+"/"
"""
getScoreDifferences(filepath + "bow/names/50-10-geonames.txt", filepath + "ndcg/placetypes mds E2000 DS[100] DN0.6 CTgeonames HAtanh CV1 S0 DevFalse LETrue SFT0L050.txt",
                    filepath + "bow/names/50-10-all.txt", filepath + "ndcg/placetypes mds E2000 DS[100] DN0.6 CTgeonames HAtanh CV1 S0 DevFalse LEFalse SFT0L050.txt",
                    "placetypes mds E3000 DS[100] DN0.6 CTgeonames HAtanh CV1 S0 DevFalse SFT0L0 LE", data_type)
print("done doing score diff")
"""
def convertToPPMIOld(freq_arrays_fn, term_names_fn):
    file = open(freq_arrays_fn)
    for line in file:
        print((len(line.split())))
    freq_arrays = np.asarray(import2dArray(freq_arrays_fn, "s"))
    term_names = import1dArray(term_names_fn)
    ppmi_arrays = []
    overall = 0.0
    for f in freq_arrays:
        overall += sum(f)
    entity_array = [0] * 15000
    # For each term
    for t in range(len(freq_arrays)):
        ppmi_array = []
        term = sum(freq_arrays[t, :])
        term_p = 0.0
        for f in freq_arrays[t, :]:
            term_p += f / overall
        for e in range(len(freq_arrays[t])):
            ppmi = 0.0
            freq = freq_arrays[t][e]
            if freq != 0:
                freq_p = freq / overall
                if entity_array[e] == 0:
                    entity = sum(freq_arrays[:, e])
                    entity_p = 0.0
                    for f in freq_arrays[:, e]:
                        entity_p += f / overall
                    entity_array[e] = entity_p
                proba = freq_p / (entity_array[e] * term_p)
                ppmi = np.amax([0.0, np.log(proba)])
            ppmi_array.append(ppmi)
        ppmi_arrays.append(ppmi_array)
        write1dArray(ppmi_array, "../data/movies/bow/ppmi/class-" + term_names[t])
    write2dArray(ppmi_arrays, "../data/movies/bow/ppmi/class-all")

def convertToPPMI(freq_arrays_fn, term_names_fn):
    freq_arrays = np.asarray(import2dArray(freq_arrays_fn, "i"))
    term_names = import1dArray(term_names_fn)
    ppmi_arrays = []
    overall = 0.0
    for f in freq_arrays:
        overall += sum(f)
    entity_array = [0] * 15000
    # For each term
    for t in range(len(freq_arrays)):
        ppmi_array = []
        term = sum(freq_arrays[t, :])
        term_p = term / overall
        for e in range(len(freq_arrays[t])):
            ppmi = 0.0
            freq = freq_arrays[t][e]
            if freq != 0:
                freq_p = freq / overall
                if entity_array[e] == 0:
                    entity = sum(freq_arrays[:, e])
                    entity_p = entity / overall
                    entity_array[e] = entity_p
                proba = freq_p / (entity_array[e] * term_p)
                ppmi = np.amax([0.0, np.log(proba)])
            ppmi_array.append(ppmi)
        print(ppmi_array)
        ppmi_arrays.append(ppmi_array)
        write1dArray(ppmi_array, "../data/movies/bow/ppmi/class-" + term_names[t])
    write2dArray(ppmi_arrays, "../data/movies/bow/ppmi/class-all")

#write1dArray(list(range(50000)), "../data/sentiment/nnet/spaces/entitynames.txt")


original_ppmi = "../data/newsgroups/bow/names/simple_numeric_stopwords_words 29-0.999-all.txt"
library_ppmi = "../data/newsgroups/bow/names/30-18836-all.txt"

#getDifference(original_ppmi, library_ppmi)

import random
"""
#Going to just use dropout instead
def saltAndPepper(movie_vectors_fn, chance_to_set_noise, salt, filename):
    movie_vectors = import2dArray(movie_vectors_fn)
    amount_to_noise = len(movie_vectors_fn) * chance_to_set_noise
    for m in range(len(movie_vectors)):
        for a in range(amount_to_noise):
            ri = random.choice(list(enumerate(movie_vectors[m])))
            if salt is True:
                movie_vectors[m][ri] = 0
            else:
                movie_vectors[m][ri] = 1
        if salt is True:
            filename += "SPN0NC" + str(chance_to_set_noise)
        else:
            filename += "SPN1NC" + str(chance_to_set_noise)
    write2dArray(movie_vectors, filename)

movie_vectors_fn = "../data/movies/bow/ppmi/class-all-normalized--1,1"

saltAndPepper(movie_vectors_fn, 0.5, True, "../data/movies/bow/ppmi/class-all-normalized--1,1")
"""

def convertPPMI_original(mat):
    """
    Compute the PPMI values for the raw co-occurrence matrix.
    PPMI values will be written to mat and it will get overwritten.
    """
    (nrows, ncols) = mat.shape
    colTotals = np.zeros(ncols, dtype="float")
    for j in range(0, ncols):
        colTotals[j] = np.sum(mat[:,j].data)
    print(colTotals)
    N = np.sum(colTotals)
    for i in range(0, nrows):
        row = mat[i,:]
        rowTotal = np.sum(row.data)
        for j in row.indices:
            val = np.log((mat[i,j] * N) / (rowTotal * colTotals[j]))
            mat[i, j] = max(0, val)
    return mat
#write2dArray(convertPPMI_original( np.asarray(import2dArray("../data/movies/bow/frequency/phrases/class-all"))), "../data/movies/bow/ppmi/class-all-lori")


def writeIndividualClasses(overall_class_fn, names_fn, output_filename):
    overall_class = import2dArray(overall_class_fn, "f")
    names = import1dArray(names_fn)
    for n in range(len(names)):
        write1dArray(overall_class[n], output_filename + "class-" + names[n])
        print(names[n])

#writeIndividualClasses("../data/movies/bow/frequency/phrases/class-all-scaled0,1.txt", "../data/movies/bow/phrase_names.txt", "../data/movies/bow/normalized_frequency/")
#writeIndividualClasses("../data/movies/bow/ppmi/class-all-scaled0,1", "../data/movies/bow/phrase_names.txt", "../data/movies/bow/normalized_ppmi/")

#plotSpace("../data/movies/nnet/spaces/films200L1100N0.5TermFrequencyN0.5FT.txt")

def getNamesFromDict(dict_fn, file_name):
    new_dict = import2dArray(dict_fn, "s")
    names = []
    for d in range(len(new_dict)):
        names.append(new_dict[d][0].strip())
    write1dArray(names, "../data/movies/cluster/hierarchy_names/" + file_name + ".txt")

#getNamesFromDict("../data/movies/cluster/hierarchy_dict/films200L1100N0.50.8.txt", "films200L1100N0.50.8.txt")



file_name = "../data/movies/finetune/films200L1100N0.5TermFrequency"




def getTop10Clusters(file_name, ids):
    clusters = np.asarray(import2dArray("../data/movies/rank/discrete/" + file_name + "P1.txt", "s")).transpose()
    cluster_names = import2dArray("../data/movies/cluster/hierarchy_names/" + file_name + "0.8400.txt", "s")
    for c in range(len(cluster_names)):
        cluster_names[c] = cluster_names[c][0]
    to_get = []
    for i in ids:
        for v in range(len(clusters[i])):
            rank = int(clusters[i][v][:-1])
            if rank <= 3:
                print(cluster_names[v][6:])
        print("----------------------")

from sklearn import svm
from sklearn.metrics import cohen_kappa_score
def obtainKappaOnClusteredDirection(names, ranks):
    # For each discrete rank, obtain the Kappa score compared to the word occ
    kappas = np.empty(len(names))
    for n in range(len(names)):
        clf = svm.LinearSVC()
        ppmi = np.asarray(import1dArray("../data/movies/bow/binary/phrases/" + names[n], "i"))
        clf.fit(ranks, ppmi)
        y_pred = clf.predict(ranks)
        score = cohen_kappa_score(ppmi, y_pred)
        kappas[n] = score
    return kappas

# Takes as input a folder with a series of files
def concatenateDirections(folder_name):
    files = getFns(folder_name)
    all_directions = []
    all_names = []
    all_kappa = []
    for f in files:
        file = open(f)
        lines = file.readlines()
        all_directions.append(lines[0])
        all_names.append(f[:5])



#getNonZero("../data/movies/classify/genres/names.txt", "../data/movies/classify/genres/class-all")



def averageCSVs(csv_array_fns):
    csv_array = []
    i = 0
    for csv_name in csv_array_fns:
        print(i, csv_name)
        i = i + 1
        csv_array.append(read_csv(csv_name))
    for csv in range(1, len(csv_array)):
        for col in range(1, len(csv_array[csv])):
            for val in range(len(csv_array[csv].iloc[col])):
                csv_array[0].iloc[col][val] += csv_array[csv].iloc[col][val]
                if np.isnan(csv_array[0].iloc[col][val] ):
                    print("NAN",  csv, col, val)
    if len(csv_array) != 0:
        for col in range(1, len(csv_array[0])):
            for val in range(len(csv_array[0].iloc[col])):
                print(csv_array[0].iloc[col][val])
                csv_array[0].iloc[col][val] = csv_array[0].iloc[col][val] / len(csv_array)
                print(csv_array[0].iloc[col][val])
                if np.isnan(csv_array[0].iloc[col][val] ):
                    print("NAN", csv_array[0].iloc[col][val], col, val)
        avg_fn = csv_array_fns[0][:len(csv_array_fns[0])-4] + "AVG.csv"
        csv_array[0].to_csv(avg_fn)
    else:
        print("FAILED CSV")
        avg_fn = "fail"
    return avg_fn

def removeCSVText(filename):
    original_fn = filename

    filename = filename.split()

    done = False
    for s in range(len(filename)):
        for i in range(10):
            if "S" +str(i) in filename[s]:
                del filename[s]
                done=True
                break
        if done:
            break
    filename = " ".join(filename)
    return original_fn, filename
"""
string1 = "places mds 100E2000DS[100]DN0.5CTopencycHAtanhCV5 S2OA sigmoid SFT0 allL050kappa KMeans CA200 MC1 MS0.4 ATS1000 DS400 tdev3RPFT BOC300.csv"
string2 = "places mds 100E2000DS[100]DN0.5CTopencycHAtanhCV5 S3OA sigmoid SFT0 allL050kappa KMeans CA200 MC1 MS0.4 ATS1000 DS400 tdev3RPFT BOC300.csv"
og_st1, st1 = removeCSVText(string1)
og_st2, st2 = removeCSVText(string2)
findDifference(st1, st2)
"""

def getScores(names, full_scores, full_names, file_name, data_type):
    full_scores = import1dArray(full_scores)
    full_names = import1dArray(full_names)
    names = import1dArray(names)
    final_scores = []
    for j in range(len(names)):
        for i in range(len(full_names)):
            if names[j] == full_names[i]:
                final_scores.append(full_scores[i])
                break
    write1dArray(final_scores, "../data/" + data_type + "/bow/scores/" + file_name + ".txt")
    return "../data/" + data_type + "/bow/scores/" + file_name + ".txt"


def reObtainName(loc, name):
    print(name)

def chunks(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]
import time
def compileSVMResults(file_name, chunk_amt, data_type):
    if fileExists("../data/"+data_type+"/svm/directions/"+file_name+".txt") is False:
        print("Compiling SVM results")
        randomcount = 0
        directions = []
        for c in range(chunk_amt):
            directions.append("../data/"+data_type+"/svm/directions/"+file_name + " CID" + str(c) + " CAMT" + str(chunk_amt)+".txt")
        kappa = []
        for c in range(chunk_amt):
            kappa.append("../data/"+data_type+"/svm/kappa/"+file_name + " CID" + str(c) + " CAMT" + str(chunk_amt)+".txt")
        for f in directions:
            while not fileExists(f):
                time.sleep(10)
        time.sleep(10)
        di = []
        for d in directions:
            di.extend(import2dArray(d))
        ka = []
        for k in kappa:
            ka.extend(import1dArray(k))
        write2dArray(di, "../data/" + data_type + "/svm/directions/" + file_name + ".txt")
        write1dArray(ka, "../data/" + data_type + "/svm/kappa/" + file_name + ".txt")
    else:
        print ("Skipping compile")


def match_entities(t_names, names):
    amount_found = 0
    for n in range(len(names)):
        names[n] = removeEverythingFromString(names[n])
    for n in range(len(t_names)):
        t_names[n] = removeEverythingFromString(t_names[n])
    matched_ids = []
    for n in range(len(t_names)):
        for ni in range(len(names)):
            matched_name = t_names[n]
            all_name = names[ni]
            if matched_name == all_name:
                matched_ids.append(ni)
                amount_found += 1
                break
    return matched_ids

def arrangeByScore(csv_fns, arra_name):
    csv_array = []
    counter = 0
    inds_to_del = []
    for csv_name in csv_fns:
        print(counter)
        try:
            csv_array.append(read_csv(csv_name).as_matrix())
        except FileNotFoundError:
            inds_to_del.append(counter)
            print("Didn't find one")
        counter = counter + 1

    csv_fns = np.delete(csv_fns, inds_to_del, axis=0)
    # Get rows of averages
    row = 0
    for c in range(len(csv_fns)):
        split = csv_fns[c].split("/")
        try:
            csv_fns[c] = split[len(split)-1]
        except IndexError:
            print("FAIL", split, csv_fns[c])

    col_names = ["1 ACC D3", "2 F1 D3","3 ACC DN", "4 F1 DN","5 ACC J48", "6 F1 J48",  "7 MICRO F1 D3",  "8 MICRO F1 DN", "9 MICRO F1 J48"]
    average_rows = []
    for csv in range(0, len(csv_array)):
        row = []
        count = 0
        for col in range(len(csv_array[csv])-2, len(csv_array[csv])):
            if count != 1:
                row.extend(csv_array[csv][col][:6])
            else:
                new_val = np.unique(np.asarray(csv_array[csv][col]))
                new_val = new_val[np.nonzero(new_val)]
                row.extend(new_val.tolist()[:3])
            count+=1
        while len(row) != 9:
            row.append(0.0)
        average_rows.append(row)
    average_rows = np.asarray(average_rows).transpose()
    write_csv(arra_name, col_names, average_rows, csv_fns)
    return 0,0,0
    print("x")

#write1dArray(list(range(20000)), "../data/sentiment/nnet/spaces/entitynames.txt")

def countClassFrequences(data_type, class_name):
    class_all = import2dArray("../data/" + data_type + "/classify/" + class_name + "/class-all")
    class_names = import1dArray("../data/" + data_type + "/classify/" + class_name + "/names.txt")
    counts = []
    class_all = np.asarray(class_all).transpose()
    for i in range(len(class_all)):
        count = len(np.nonzero(class_all[i])[0])
        print(class_names[i], count)
        counts.append(count)

def removeInfrequent(classes, class_names, remove_all_classes_below):
    infrequent_classes = []
    if len(classes) > len(classes[0]):
        classes = np.asarray(classes).transpose()
    for i in range(len(classes)):
        count = len(np.nonzero(classes[i])[0])
        if count < remove_all_classes_below:
            infrequent_classes.append(i)
    classes = np.delete(classes, infrequent_classes, axis=0)
    class_names = np.delete(class_names, infrequent_classes, axis=0)
    print("deleted", len(infrequent_classes), "classes now", len(class_names), "classes")
    return np.asarray(classes.transpose(), dtype=np.int32), class_names



"""

"""
if __name__ == '__main__':
    """
    #countClassFrequences("reuters", "topics")
    class_fn = "../data/movies/classify/keywords/class-all"
    class_name_fn = "../data/movies/classify/keywords/names.txt"
    classes = import2dArray(class_fn)
    class_names = import1dArray(class_name_fn)
    classes, class_names = removeInfrequent(classes, class_names)
    """
    words = import1dArray("../data/placetypes/bow/names/5-1-all.txt", "s")
    word_dict = {}
    for i in range(len(words)):
        word_dict[i] = words[i]

    averageWordVectorsFreq(word_dict,
                           "../data/placetypes/bow/frequency/phrases/class-all-5-1-all",
                           200,
                           "placetypes")
    averageWordVectorsFreq(word_dict,
                           "../data/placetypes/bow/frequency/phrases/class-all-5-1-all",
                           100,
                           "placetypes")
    averageWordVectorsFreq(word_dict,
                           "../data/placetypes/bow/frequency/phrases/class-all-5-1-all",
                           50,
                           "placetypes")

    """

    averageWordVectors(word_dict,
                           "../data/movies/bow/ppmi/class-all-25-5-genres",
                           200,
                           "newsgroups")
    averageWordVectorsFreq("../data/raw/sentiment/simple_numeric_stopwords_vocab 2.npy",
                           "../data/sentiment/bow/frequency/phrases/simple_numeric_stopwords_bow 2-all.npz",
                           200,
                           "sentiment")

    averageWordVectorsFreq("../data/raw/sentiment/simple_numeric_stopwords_vocab 2.npy",
                           "../data/sentiment/bow/frequency/phrases/simple_numeric_stopwords_bow 2-all.npz",
                           50,
                           "sentiment")
    averageWordVectorsFreq("../data/raw/sentiment/simple_numeric_stopwords_vocab 2.npy",
                           "../data/sentiment/bow/frequency/phrases/simple_numeric_stopwords_bow 2-all.npz",
                           100,
                           "sentiment")

    name = "../data/newsgroups/nnet/spaces/simple_numeric_stopwords_ppmi 2-all_mds50.txt"
    write2dArray(import2dArray(name, "f").transpose(), name)
    """

    """
    data_type = "movies"
    representation = "../data/raw/previous work/filmids.txt"
    representation = import1dArray(representation, "i")

    inds_to_del = []
    for i in range(len(representation)):
        if representation[i] == -1:
            inds_to_del.append(i)

    representation = np.delete(representation, inds_to_del)

    print(len(representation))

    #mds1 = "../data/movies/nnet/spaces/wvFIXED200.npy"
    #mds1 = import2dArray(mds1, "f")
    mds2 = "../data/movies/nnet/spaces/wvFIXED100.npy"
    mds2 = import2dArray(mds2, "f")
    mds3 = "../data/movies/nnet/spaces/wvFIXED50.npy"
    mds3 = import2dArray(mds3, "f")
    if len(mds2) == 15000:
        #mds1 = np.delete(mds2, inds_to_del)
        mds2 = np.delete(mds2, inds_to_del)
        mds3 = np.delete(mds3, inds_to_del)

    print(len(mds2))
    representation, inds = np.unique(representation, return_index=True)
    print(len(representation))
    print(len(inds))


    #mds1 = mds2[inds]
    mds2 = mds2[inds]
    mds3 = mds3[inds]
    print(len(mds2))
    #write2dArray(mds1, "../data/movies/nnet/spaces/wvFIXED200.npy")
    write2dArray(mds2, "../data/movies/nnet/spaces/wvFIXED100.npy")
    write2dArray(mds3, "../data/movies/nnet/spaces/wvFIXED50.npy")

    """
    """
    main_names_fn = "../data/movies/nnet/spaces/entitynames.txt"
    main_names = import1dArray(main_names_fn, "s")
    rating_names_fn = "../data/movies/classify/ratings/available_entities.txt"
    rating_names = import1dArray(rating_names_fn, "s")
    # Get IDS of entities that are duplicates or -1
    representation = "../data/raw/previous work/filmids.txt"
    representation = import1dArray(representation, "i")
    classes_fn = "../data/movies/classify/ratings/class-All"
    classes = import2dArray(classes_fn, "i")

    inds_to_del = []
    for i in range(len(representation)):
        if representation[i] == -1:
            inds_to_del.append(i)

    representation = np.delete(representation, inds_to_del)

    print(len(representation))
    names_del = main_names[inds_to_del]
    if len(main_names):
        main_names = np.delete(main_names, inds_to_del)

    print(len(main_names))
    ns, inds, counts = np.unique(representation, return_index=True, return_counts=True)
    print(len(representation))
    print(len(inds))

    duplicate_inds = np.delete(list(range(len(representation))), inds)

    names_to_remove = main_names[duplicate_inds]

    matching_ids = match_entities(rating_names, names_to_remove)

    rating_names = np.delete(rating_names, matching_ids)
    classes = np.delete(classes, matching_ids)

    write2dArray(classes, classes_fn)
    write1dArray(rating_names, rating_names_fn)

    # Get names of entities that are duplicates or -1

    # Get ids from the ratings names corresponding to these entities

    # Remove these ids from the classes and names of ratings

    # Remove these ids from the overall entitynames
    """



"""
bow_fn = "../data/movies/bow/ppmi/class-all-100-10-all"
bow = import2dArray(bow_fn, "f").transpose()
print(len(bow))
bow = bow[inds]
print(len(bow))
bow = sp.csr_matrix(bow)
sp.save_npz(bow_fn + "-nodupe.npz", bow.transpose())

bow_fn = "../data/movies/bow/frequency/phrases/class-all-100-10-all"
bow = import2dArray(bow_fn, "i").transpose()
print(len(bow))
bow = bow[inds]
print(len(bow))
bow = sp.csr_matrix(bow)
sp.save_npz(bow_fn + "-nodupe.npz", bow.transpose())

"""

"""
id_1 = np.load("../data/raw/newsgroups/simple_remove.npy")
id_2 = np.load("../data/raw/newsgroups/simple_stopwords_remove.npy")
representation = import2dArray("../data/newsgroups/nnet/spaces/fastText E300 ML200 MF158248 E20 NG1 PRETrue.npy")

representation = np.delete(representation, id_1, axis=0)
representation = np.delete(representation, id_2, axis=0)

np.save("../data/newsgroups/nnet/spaces/fastText E300 ML200 MF158248 E20 NG1 PRETrue.npy", representation)
"""
"""
representation = import2dArray("../data/newsgroups/bow/frequency/phrases/simple_stopwords_bow 2-gram50-0.99-all.npz")
print("hi")

lines = import2dArray("../data/output.txt", "s")

kappa = []
f1 = []
acc = []
for i in range(len(lines)):
    kappa.append(lines[i][4])
    f1.append(lines[i][6])
    acc.append(lines[i][8])

file_name = "fastTextCV1S0 SFT0 allL03018836 LR "

st = "../data/newsgroups/svm/"
write1dArray(kappa, st + "kappa/" + file_name)
write1dArray(acc, st + "acc/" + file_name)
write1dArray(f1, st + "f1/" + file_name)
"""
"""
representation = np.load("../data/newsgroups/nnet/spaces/MF5000 ML200 BS32 FBTrue DO0.3 RDO0.05 E64 ES16LS32 L1.txt.npy")

write2dArray(representation, "../data/sentiment/nnet/spaces/5kdefaultsentDEV.txt")
"""

""" #REVERSAL """
"""
fns = ["all-100-10DTP0.1TWP0.001NT400", "all-100-10DTP0.1TWP0.01NT100", "all-100-10DTP0.1TWP0.001NT400"]

for f in fns:
    full_fn = "../data/movies/LDA/names/" + f + ".txt"
    a = import2dArray(full_fn, "s")
    for i in range(len(a)):
        a[i] = np.flipud(a[i])
    write2dArray(a, full_fn)
"""

"""
fns = getFns("../data/movies/classify/keywords/")
counts = []
for fn in fns:
    blob = import1dArray("../data/movies/classify/keywords/"  + fn)
    count = 0
    for i in blob:
        if i == 1:
            count+=1
    counts.append(count)

ids = np.argsort(counts)
ids = reversed(ids)
for id in ids:
    print(fns[id])
"""
"""
write2dArray(deleteAllButIndexes(import2dArray("../data/movies/cluster/hierarchy_directions/films200-genres100ndcg0.9200.txt", "s"),
                                               import1dArray("../data/movies/cluster/hierarchy_names/human_ids films200genres.txt")),
                                 "../data/movies/cluster/hierarchy_directions/films200-genres100ndcg0.9200 human_prune.txt")

"""
#getTop10Clusters("films100L2100N0.5", [1644,164,4018,6390])

"""
from sklearn.datasets import dump_svmlight_file
genre_names = import1dArray("../data/movies/classify/genres/names.txt", "s")


genres = np.asarray(import2dArray("../data/movies/classify/genres/class-all", "i")).transpose()

class_all = []
#for i in range(23):
    #g = import1dArray("../data/movies/classify/genres/class-" + genre_names[i], "i")
    #class_all.append(g)

class_all = np.asarray(class_all).transpose()

#write2dArray(class_all, "../data/movies/classify/genres/class-all")

space_name = "films200-genres100ndcg0.85200 tdev3004FTL0 E100 DS[200] DN0.5 CTgenres HAtanh CV1 S0 DevFalse SFT0L0100ndcg0.95200MC1"
representation = np.asarray(import2dArray("../data/movies/rank/numeric/"+space_name+".txt")).transpose()

writeArff(representation, genres, genre_names, "../data/movies/keel/vectors/"+space_name+"genres", header=True)

#np.savetxt( "../data/movies/keel/vectors/"+space_name+"np.csv", representation, delimiter=",")
"""



"""



fn = "films200L325N0.5"
cluster_names_fn = "../data/movies/cluster/names/" + fn + ".txt"
file_name = fn + "InClusterN0.5FTadagradcategorical_crossentropy100"
cluster_vectors_fn = "../data/movies/nnet/clusters/" + file_name + ".txt"
new_v = []
new_v.append(import1dArray(cluster_names_fn))

fn = "films200L250N0.5"
cluster_names_fn = "../data/movies/cluster/names/" + fn + ".txt"
file_name = fn + "InClusterN0.5FTadagradcategorical_crossentropy100"
cluster_vectors_fn = "../data/movies/nnet/clusters/" + file_name + ".txt"

new_v.append(import1dArray(cluster_names_fn))

fn = "films200L1100N0.5"
cluster_names_fn = "../data/movies/cluster/names/" + fn + ".txt"
file_name = fn + "InClusterN0.5FTadagradcategorical_crossentropy100"
cluster_vectors_fn = "../data/movies/nnet/clusters/" + file_name + ".txt"

new_v.append(import1dArray(cluster_names_fn))

concatenateArrays(new_v, cluster_names_fn+"ALL")
#representation = import2dArray(file_name + ".txt")

#scaleSpaceUnitVector(representation, file_name+"uvscaled.txt")

#scaleSpace(representation, 0, 1, file_name +"scaled")
"""
"""
file = open("../data/movies/bow/ppmi/class-all-normalized--1,1")

for line in file:
    line = line.split()
    for l in range(len(line)):
        line[l] = float(line[l])
        if line[l] > 1 or line[l] < -1:
            print("FAILED!", line[l])
    print(line)

plotSpace(scaleSpace(import2dArray("../data/movies/bow/ppmi/class-all"), -1, 1, "../data/movies/bow/ppmi/class-all-normalized--1,1"))
"""
#convertToTfIDF("../data/movies/bow/frequency/phrases/class-All")
#convertToPPMI("../data/movies/bow/frequency/phrases/class-All", "../data/movies/bow/phrase_names.txt")

"""
file = np.asarray(import2dArray("../data/movies/bow/tfidf/class-All")).transpose()
phrase_names = import1dArray("../data/movies/bow/phrase_names.txt")
movie_names = import1dArray("../data/movies/nnet/spaces/filmNames.txt")
example = file[1644]
indexes = np.argsort(example)
for i in indexes:
    print(phrase_names[i])
"""
"""
file_name = "all results mds"
arrange_name = "all results mds"
all_csv_fns = []
loc = "../data/" + data_type + "/rules/tree_csv/"
fns_to_add = getCSVsToAverage(loc)
for f in range(len(fns_to_add)):
    fns_to_add[f] = "../data/" + data_type + "/rules/tree_csv/" + fns_to_add[f]
all_csv_fns.extend(fns_to_add)
arrangeByScore(np.unique(np.asarray(all_csv_fns)), loc   + " " + arrange_name + file_name[:50] + str(len(all_csv_fns)) + ".csv")
"""