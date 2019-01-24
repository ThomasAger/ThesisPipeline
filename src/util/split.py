# Train splits defined by various standards
imdb_train = 25000
newsgroups_train = 11314
reuters_train = 7656
yahoo_train = 25000
movies_train = 9225
placetypes_train = 913

imdb_total = 50000
newsgroups_total = 18846
reuters_total = 10655
yahoo_total = 50000
movies_total = 13978
placetypes_total = 1383



from sklearn.model_selection import KFold
import numbers
from util import py as pu
from util import py


def get_name_dict(*params):
    print(*params)


def get_doc_amt(data_type):
    if data_type == "imdb" or data_type == "sentiment":
        max_size = imdb_total
    elif data_type == "newsgroups":
        max_size = newsgroups_total
    elif data_type == "reuters":
        max_size = reuters_total
    elif data_type == "yahoo" or data_type == "amazon":
        max_size = yahoo_total
    elif data_type == "movies":
        max_size = movies_total
    elif data_type == "placetypes":
        max_size = placetypes_total
    else:
        print("No data type found")
        raise ValueError("Data type not found", data_type)
    return max_size


def check_shape(features, data_type):
    print("Shape", len(features), len(features[0]))
    if data_type == "imdb" or data_type == "sentiment":
        max_size = imdb_total
    elif data_type == "newsgroups":
        max_size = newsgroups_total
    elif data_type == "reuters":
        max_size = reuters_total
    elif data_type == "yahoo" or data_type == "amazon":
        max_size = yahoo_total
    elif data_type == "movies":
        max_size = movies_total
    elif data_type == "placetypes":
        max_size = placetypes_total
    else:
        print("No data type found")
        raise ValueError("Data type not found", data_type)
    if len(features) != max_size and len(features[0]) != max_size:
        raise ValueError(print(len(features), "This is not the standard size, expected " + str(max_size)))
    return True

def get_split_ids(data_type, matched_ids):
    # Multiple data-type names in-case im stupid
    if data_type == "imdb" or data_type == "sentiment":
        train_split = imdb_train
        total = imdb_total
    elif data_type == "newsgroups":
        train_split = newsgroups_train
        total = newsgroups_total
    elif data_type == "reuters":
        train_split = reuters_train
        total = reuters_total
    elif data_type == "yahoo" or data_type == "amazon":
        train_split = yahoo_train
        total = yahoo_total
    elif data_type == "movies":
        train_split = movies_train
        total = movies_total
    elif data_type == "placetypes":
        train_split = placetypes_train
        total = placetypes_total
    else:
        print("No data type found")
        return False
    if matched_ids is None:
        ids = list(range(total))
        x_train_split = train_split
        y_train = ids[:x_train_split]
        y_test = ids[x_train_split:]
    else:
        print("Matched ids of len", len(matched_ids), "instead of", total)
        ids = matched_ids
        x_train_split = int(len(ids) * 0.66)
        y_ids = list(range(len(ids)))
        y_train_split = int(len(ids) * 0.66)
        y_train = y_ids[:y_train_split]
        y_test = y_ids[y_train_split:]
    x_train = ids[:x_train_split]
    x_test = ids[x_train_split:]

    print("before dev split")
    print(len(x_train),  "test")
    print(len(x_test), "train")

    if len(x_train) != len(y_train) or len(x_test) != len(y_test):
        raise ValueError("Ids not same length.")

    return {"x_train":x_train, "x_test":x_test, "y_train":y_train, "y_test":y_test}

import numpy as np

def split_data(x, y, split_ids, dev_percent_of_train=0.2):
    y = py.transIfColsLarger(y)
    x_len = len(x)
    y_len = len(y)
    x_train = x[split_ids["x_train"]]
    y_train = y[split_ids["y_train"]]
    x_test = x[split_ids["x_test"]]
    y_test = y[split_ids["y_test"]]
    if dev_percent_of_train > 0:
        x_dev = x_train[int(len(x_train) * (1 - dev_percent_of_train)):]
        y_dev = y_train[int(len(y_train) * (1 - dev_percent_of_train)):]
        x_train = x_train[:int(len(x_train) * (1 - dev_percent_of_train))]
        y_train = y_train[:int(len(y_train) * (1 - dev_percent_of_train))]
        if np.amax(x_train) == np.amax(y_train) and np.amax(x_test) == np.amax(y_test):
            if (len(x_train) + len(x_test) + len(x_dev)) != x_len:
                raise ValueError(str(x_len) + "does not equal its components")
            if (len(y_train) + len(y_test) + len(y_dev)) != y_len:
                raise ValueError(str(y_len)+ "does not equal its components")

    else:
        if np.amax(x_train) == np.amax(y_train) and np.amax(x_test) == np.amax(y_test):
            if (len(x_train) + len(x_test)) != x_len:
                raise ValueError(str(x_len) + "does not equal its components")
            if (len(y_train) + len(y_test)) != y_len:
                raise ValueError(str(y_len)+ "does not equal its components")
    if len(x_dev) > len(x_train) and dev_percent_of_train <= 0.5:
        raise ValueError("Dev split is larger than train split")
    return x_train, y_train, x_test, y_test, x_dev, y_dev

def old_split(features, classes, data_type, dev_percent=0.2):
    check_shape(features, data_type)
    if data_type == "imdb" or data_type == "sentiment":
        train_split = imdb_train
    elif data_type == "newsgroups":
        train_split = newsgroups_train
    elif data_type == "reuters":
        train_split = reuters_train
    elif data_type == "yahoo" or data_type == "amazon":
        train_split = yahoo_train
    elif data_type == "movies":
        train_split = movies_train
    elif data_type == "placetypes":
        train_split = placetypes_train
    else:
        print("No data type found")
        return False

    x_train = features[:train_split]
    x_test = features[train_split:]
    y_train = classes[:train_split]
    y_test = classes[train_split:]

    x_dev = None
    y_dev = None
    if dev_percent > 0:
        x_dev = x_train[int(len(x_train) * (1 - dev_percent)):]
        y_dev = y_train[int(len(y_train) * (1 - dev_percent)):]
        x_train = x_train[:int(len(x_train) * (1 - dev_percent))]
        y_train = y_train[:int(len(y_train) * (1 - dev_percent))]
        print(len(x_dev), len(x_dev[0]), "x_dev")
        print(len(y_dev),  "y_dev")

    print(len(x_test), len(x_test[0]), "x_test")
    print(len(y_test),  "y_test")
    print(len(x_train), len(x_train[0]), "x_train")
    print(len(y_train),  "y_train")

    check_splits(x_train, y_train, x_test, y_test)

    return x_train, y_train, x_test, y_test, x_dev, y_dev




def crossValData(cv_splits, features, classes):
    ac_y_train = []
    ac_x_train = []
    ac_x_test = []
    ac_y_test = []
    ac_y_dev = []
    ac_x_dev = []
    kf = KFold(n_splits=cv_splits, shuffle=False, random_state=None)
    for train, test in kf.split(features):
        ac_y_test.append(classes[test])
        ac_y_train.append(classes[train[:int(len(train) * 0.8)]])
        ac_x_train.append(features[train[:int(len(train) * 0.8)]])
        ac_x_test.append(features[test])
        ac_x_dev.append(features[train[int(len(train) * 0.8):len(train)]])
        ac_y_dev.append(classes[train[int(len(train) * 0.8):len(train)]])
        c += 1
    return ac_x_train, ac_y_train, ac_x_test, ac_y_test, ac_x_dev, ac_y_dev

