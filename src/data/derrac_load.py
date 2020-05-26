from util import io as dt
import numpy as np
def cleanIds(ids):
    print("Ids, len", len(ids))
    ids_to_remove = []
    for i in range(len(ids)):
        if ids[i] == -1:
            ids_to_remove.append(i)
    ids = np.delete(ids, ids_to_remove)
    print("Ids, -1 removed, len", len(ids))
    ids = np.unique(ids)
    print("Ids, non-unique removed, len", len(ids))
    return ids


def processMovies():
    data_type = "movies"
    orig_fn = "../../data/raw/" + data_type + "/"
    raw_fn = "../../data/raw/derrac/"+ data_type + "/"
    ids = dt.import1dArray(orig_fn + "filmIdsClean.txt", "i")
    names = dt.import1dArray(orig_fn + "filmNamesClean.txt", "s")
    text_corpus = []
    for i in range(len(ids)):
        print(names[i], i, "/", len(ids))
        corpus_string = ""
        lines = dt.import1dArray(raw_fn + "tokens/" + str(ids[i]) + ".film")
        for z in range(len(lines)):
            to_add_string = ""
            if "#" in lines[z]:
                continue
            split_line = lines[z].split()
            for j in range(int(split_line[1])):
                to_add_string += split_line[0] + " "
            corpus_string += to_add_string
        text_corpus.append(corpus_string)
    dt.write1dArray(text_corpus, "../../data/raw/movies/corpus.txt")




def processPlacetypes():
    data_type = "placetypes"
    orig_fn = "../../data/raw/derrac/" + data_type + "/"
    ids = dt.import1dArray(orig_fn + "placeNames.txt", "s")
    text_corpus = []
    for i in range(len(ids)):
        print(ids[i], i, "/", len(ids))
        corpus_string = ""
        lines = dt.import1dArray(orig_fn + "tokens/" + str(ids[i]) + ".photos")
        for z in range(len(lines)):
            to_add_string = ""
            if "#" in lines[z]:
                continue
            split_line = lines[z].split()
            for j in range(int(split_line[1])):
                to_add_string += split_line[0] + " "
            corpus_string += to_add_string
        text_corpus.append(corpus_string)
    dt.write1dArray(text_corpus, "../../data/raw/placetypes/corpus.txt")
import string
from data import process_corpus
# Matching the ids between the original list of film names and the duplicate list of film names
def match_ids_duplicates():
    data_type = "movies"
    orig_fn = "../../data/raw/" + data_type + "/"
    raw_fn = "../../data/raw/derrac/"+ data_type + "/"
    names_orig = dt.import1dArray(raw_fn + "filmNames.txt", "s")
    names_clean = dt.import1dArray(orig_fn + "filmNamesClean.txt", "s")

    print("Cleaning names")
    # Ensure the names are formatted the same
    names_orig = process_corpus.preprocess(names_orig)
    names_clean = process_corpus.preprocess(names_clean)
    # Removing weird characters that appeared when pre-processing using gensim
    for i in range(len(names_orig)):
        names_orig[i] = names_orig[i].replace(" ", "").replace("¡", "").replace("¶", "").replace("”","")
    print("Cleaned, sample", np.random.choice(names_orig))
    print("Compare to", np.random.choice(names_clean))
    # Match them
    ids = []
    didnt_find = []
    print("Len of clean names is", len(names_clean))
    for i in range(len(names_clean)):
        for j in range(len(names_orig)):
            if names_orig[j] == names_clean[i]:
                ids.append(j)
                break
            if j == len(names_orig) - 1:
                didnt_find.append(names_clean[i])
                print("Didn't find", i, didnt_find)
        if i % 1000 == 0:
            print(i, "/", len(names_clean))

    if len(didnt_find) == 0:
        print("Found all", len(names_clean))

    np.save(orig_fn + "filmIdsNoDuplicates.npy", ids)

def get_only_ids():
    data_type = "movies"
    orig_fn = "../../data/raw/" + data_type + "/"
    c_fn_start = "../../data/raw\Data_NeSy16\Input Vectors/"
    use_cluster_fn = [c_fn_start + "L0.npy", c_fn_start + "L1.npy", c_fn_start + "L2.npy", c_fn_start + "L3.npy",
                      c_fn_start + "L4.npy"]
    ids = np.load(orig_fn + "filmIdsNoDuplicates.npy")

    for i in range(len(use_cluster_fn)):
        id_array = np.load(use_cluster_fn[i])[ids]
        np.save(use_cluster_fn[i][:-4] + "_nodupe.npy", id_array)
        print(use_cluster_fn[i][:-4] + "_nodupe.npy")

def combine_spaces():
    c_fn_start = "../../data/raw\Data_NeSy16\Input Vectors/"
    use_cluster_fn = [c_fn_start + "L0_nodupe.npy", c_fn_start + "L1_nodupe.npy", c_fn_start + "L2_nodupe.npy", c_fn_start + "L3_nodupe.npy", c_fn_start + "L4_nodupe.npy"]

    space_0 = np.load(use_cluster_fn[0])
    space_1 = np.load(use_cluster_fn[0])
    space_2 = np.load(use_cluster_fn[0])
    space_3 = np.load(use_cluster_fn[0])
    space_4 = np.load(use_cluster_fn[0])

    np.save(c_fn_start + "L0_and_4.npy", np.concatenate([space_0, space_4], axis=1))
    np.save(c_fn_start + "LAllConcat.npy", np.concatenate([space_0, space_1, space_2, space_3, space_4], axis=1))

def combine_clusters():
    c_name_fn_start = "../../data/raw\Data_NeSy16\Cluster Classes/"
    use_cluster_name_fn = [c_name_fn_start + "L0 Cluster.npy", c_name_fn_start + "L1 Cluster.npy", c_name_fn_start + "L2 Cluster.npy",
                           c_name_fn_start + "L3 Cluster.npy", c_name_fn_start + "L4 Cluster.npy"]

    space_0 = np.load(use_cluster_name_fn[0])
    space_1 = np.load(use_cluster_name_fn[0])
    space_2 = np.load(use_cluster_name_fn[0])
    space_3 = np.load(use_cluster_name_fn[0])
    space_4 = np.load(use_cluster_name_fn[0])

    np.save(c_name_fn_start + "L0_and_4.npy", np.concatenate([space_0, space_4], axis=0))
    np.save(c_name_fn_start + "LAllConcat.npy", np.concatenate([space_0, space_1, space_2, space_3, space_4], axis=0))

def getFreqsOfClusters(clusters, word_freq_dict, limit):

    average_freqs = []
    for i in range(len(clusters)):
        total = 0
        if limit != 0:
            for j in range(len(clusters[i][:limit])):
                total+= word_freq_dict[clusters[i][j]]
            average_freqs.append(total / len(clusters[i][:limit]))
        else:
            for j in range(len(clusters[i])):
                total+= word_freq_dict[clusters[i][j]]
            average_freqs.append(total / len(clusters[i]))



    return average_freqs

def getFreqsOfAllClusters(limit):
    data_type = "movies"
    orig_fn = "../../data/raw/" + data_type + "/"
    word_freq_dict = dt.load_dict(orig_fn + "word_freq_dict.pkl")

    c_name_fn_start = "../../data/raw\Data_NeSy16\Cluster Classes/"
    freq_start = "../../data/raw\Data_NeSy16\Cluster Freqs/"
    use_cluster_name_fn = [c_name_fn_start + "L0 Cluster.npy", c_name_fn_start + "L1 Cluster.npy",
                           c_name_fn_start + "L2 Cluster.npy",
                           c_name_fn_start + "L3 Cluster.npy", c_name_fn_start + "L4 Cluster.npy"]
    freq_fn = [freq_start + "L0 Cluster.npy", freq_start + "L1 Cluster.npy",
                           freq_start + "L2 Cluster.npy",
                           freq_start + "L3 Cluster.npy", freq_start + "L4 Cluster.npy"]

    cluster_names = []
    for i in range(len(use_cluster_name_fn)):
        cluster_names.append(np.load(use_cluster_name_fn[i]))

    all_freqs = []
    for i in range(len(cluster_names)):
        freq = getFreqsOfClusters(cluster_names[i], word_freq_dict, limit)
        all_freqs.append(freq)
        np.save(freq_fn[i], freq)
    np.save(freq_start + "all_freqs.npy", all_freqs)
    return all_freqs, freq_fn






def makeFreq():
    data_type = "movies"
    orig_fn = "../../data/raw/" + data_type + "/"
    raw_fn = "../../data/raw/derrac/" + data_type + "/"
    ids = dt.import1dArray(orig_fn + "filmIdsClean.txt", "i")
    names = dt.import1dArray(orig_fn + "filmNamesClean.txt", "s")
    word_freq_dict = {}
    for i in range(len(ids)):
        print(names[i], i, "/", len(ids))
        lines = dt.import1dArray(raw_fn + "tokens/" + str(ids[i]) + ".film")
        for z in range(len(lines)):
            split_lines = lines[z].split()
            if "#" in split_lines[0]:
                continue
            try:
                word_freq_dict[split_lines[0]] += int(split_lines[1])
            except KeyError:
                word_freq_dict[split_lines[0]] = int(split_lines[1])
    dt.save_dict(orig_fn + "word_freq_dict.pkl", word_freq_dict)

#processPlacetypes()
if __name__ == '__main__':
    limit = 1
    for i in range(7):
        limit = i
        all_freqs, names = getFreqsOfAllClusters(limit)
        all_freqs[2] = np.pad(all_freqs[2], pad_width=[0,100], mode="constant")
        all_freqs[3] = np.pad(all_freqs[3], pad_width=[0,150], mode="constant")
        all_freqs[4] = np.pad(all_freqs[4], pad_width=[0,175], mode="constant")
        col_names = [ names[0][-14:][:-4], names[1][-14:][:-4], names[2][-14:][:-4], names[3][-14:][:-4], names[4][-14:][:-4]]
        cols_to_add = [*all_freqs]
        key = None
        csv_fn = names[0][:-4] + "ALL " + str(limit) +".csv"
        dt.write_csv(csv_fn, col_names, cols_to_add, key)
