
from util import sim
from util import py
import numpy as np
def sortByScoreWordArrays(array_to_sort, word_array):
    sorted_array = []
    for i in range(len(word_array)):
        for j in range(len(array_to_sort)):
            if array_to_sort[j] == word_array[i]:
                sorted_array.append(word_array[i])

                break
    return sorted_array

def getPairs(*params):
    pairs = []
    for i in range(len(params)):
        pair = []
        pair.append(params[i])
        concat_final = np.asarray([])
        for j in range(len(params)):
            if j == i:
                continue
            else:
                concat_final = np.concatenate([concat_final, params[j]], axis=0)
        pair.append(concat_final)
        pairs.append(pair)
    return pairs



def termsUniqueOnlyTo(this_array, all_other_arrays):
    ids_to_del = []
    for i in range(len(this_array)):
        for j in range(len(all_other_arrays)):
            if this_array[i] == all_other_arrays[j]:
                ids_to_del.append(i)
    inds = list(range(len(this_array)))
    remaining_inds = np.delete(inds, ids_to_del)
    return remaining_inds

def termsCommonTo(arrays, amt_of_arrays):
    ids_to_del = []
    counts = []
    for i in range(len(arrays)):
        counts.append(arrays.count(arrays[i]))
    common_ids = []
    for i in range(len(counts)):
        if counts[i] == amt_of_arrays:
            common_ids.append(i)
    common_ids = np.unique(common_ids)
    return common_ids

import sys

def printPretty(word_array):
    for k in range(len(word_array)):
        if k == 0:
            sys.stdout.write(word_array[k] + " (")
        elif k != len(word_array) - 1:
            sys.stdout.write(word_array[k] + ", ")
        else:
            sys.stdout.write(word_array[k])
    sys.stdout.write(")\n")

def getPretty(word_array):
    word_output_array = []
    for i in range(len(word_array)):
        word_output = ""
        for k in range(len(word_array[i])):
            word_array[i][k] = str(word_array[i][k])
            if k == 0:
                word_output += word_array[i][k] + " ("
            elif k != len(word_array[i]) - 1:
                word_output += word_array[i][k] + ", "
            else:
                word_output += word_array[i][k] + ")"
        word_output_array.append(word_output)
    return word_output_array

def clusterPretty(word_array):

    word_output = "("
    for k in range(len(word_array)):
        word_array[k] = str(word_array[k])
        if k != len(word_array) - 1 or k == 0:
            word_output += word_array[k] + ", "
        else:
            word_output = word_output + word_array[k] + ")"

    return word_output

def contextualizeWords(words, word_directions, context_words, ctx_word_directions):
    word_arrays = []
    for i in range(len(words)):
        inds = sim.getXMostSimilarIndex(word_directions[i], ctx_word_directions, [], 2)
        word_arrays.append([words[i]])
        for j in inds:
            word_arrays[i].append(context_words[j])
        printPretty(word_arrays[i])
    return word_arrays

def mapWordsToContext(words, context):
    mapped_ctx = []
    amt_found = 0
    for i in range(len(words)):
        found = False
        for j in range(len(context)):
            if words[i] == context[j][0]:
                found = True
                amt_found += 1
                mapped_ctx.append(context[j])
                break
        if found is False:
            mapped_ctx.append([words[i], "N/A", "N/A"])
    print("Found", amt_found, "matches")
    if len(mapped_ctx) == 0:
        print("Incorrect inputs, ctx doesn't match words")
        exit()
    return mapped_ctx

def getPrettyStrings(array):
    pretty_strings = []
    for i in range(len(array)):
        pretty_clusters = []
        for j in range(len(array[i])):
            split = array[i][j].split()
            pretty_clusters.append(clusterPretty(split))
        cluster_string = ""
        for i in range(len(pretty_clusters)):
            cluster_string += pretty_clusters[i] + " \n "
        pretty_strings.append(cluster_string)
    return pretty_strings