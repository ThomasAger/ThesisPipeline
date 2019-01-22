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
    orig_fn = "../../data/raw/derrac/" + data_type + "/"
    ids = dt.import1dArray(orig_fn + "filmIdsClean.txt", "i")
    names = dt.import1dArray(orig_fn + "filmNamesClean.txt", "s")
    text_corpus = []
    for i in range(len(ids)):
        print(names[i], i, "/", len(ids))
        corpus_string = ""
        lines = dt.import1dArray(orig_fn + "tokens/" + str(ids[i]) + ".film")
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

processPlacetypes()