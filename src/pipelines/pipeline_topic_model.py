import util.classify
import util.text_utils
from util import io as dt
import numpy as np
# import nltk
# nltk.download()
from data import process_corpus
from util.save_load import SaveLoad
from util import split
import os
from util.text_utils import LimitWords, LimitWordsNumeric
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier, OutputCodeClassifier
from project.get_directions import GetDirections
from score.classify import MultiClassScore
from project.get_rankings import GetRankings, GetRankingsStreamed
from project.get_ndcg import GetNDCG, GetNDCGStreamed
from rep import pca, ppmi, awv
from project.get_tsr import GetTopScoringRanks, GetTopScoringRanksStreamed
from project.get_tsd import GetTopScoringDirs, GetTopScoringDirsStreamed
from util import py
from scipy import sparse as sp
import KFoldHyperParameter
from model import filter_bow
# The overarching pipeline to obtain all prerequisite data for the derrac pipeline
# Todo: Better standaradize the saving/loading
last_dct = []
from model import topicmodel

def pipeline(file_name,  bow, dct, classes, class_names, words_to_get, processed_folder, dims, kfold_hpam_dict, hpam_dict,
                     model_type="", dev_percent=0.2, rewrite_all=False, remove_stop_words=True,
                     score_metric="", auroc=False, dir_min_freq=0.001, dir_max_freq=0.95, name_of_class="", space_name="", data_type="",
             classifier_fn="", mcm=None, top_scoring_dirs=2000, score_type="kappa", ppmi=None, dct_unchanged=None, pipeline_hpam_dict=None):
    matched_ids = []
    try:
        class_entities = dt.import1dArray(processed_folder + "classes/" + name_of_class + "_entities.txt")
        entity_names = dt.import1dArray(processed_folder + "corpus/entity_names.txt")
        for i in range(len(class_entities)):
            for j in range(len(entity_names)):
                if class_entities[i] == entity_names[j]:
                    matched_ids.append(j)
                    break
    except FileNotFoundError:
        matched_ids = None

    hpam_save = SaveLoad(rewrite=True)

    # Folds and space are determined inside of the method for this hyper-parameter selection, as it is stacked
    hyper_param = KFoldHyperParameter.RecHParam(None, classes, class_names, pipeline_hpam_dict, kfold_hpam_dict, "topic",
                                                model_type,
                                                file_name, None, processed_folder + "rank/", hpam_save,
                                                probability=False,
                                                rewrite_model=rewrite_all, dev_percent=dev_percent,
                                                name_of_class=name_of_class,
                                                data_type=data_type, score_metric=score_metric, auroc=auroc,
                                                matched_ids=matched_ids, end_fn_added=name_of_class,
                                                mcm=mcm,
                                                hpam_params=[dct_unchanged, dct, bow, dir_min_freq, dir_max_freq,
                                                             file_name, processed_folder,
                                                             words_to_get,  name_of_class,  classes,
                                                             class_names, auroc, score_metric,
                                                             mcm, dev_percent, ppmi, kfold_hpam_dict, data_type,
                                                             rewrite_all, matched_ids, model_type])
    hyper_param.process_and_save()
    print("END OF SPACE")
    return hyper_param.getTopScoringRowData()


def pipeline_topic_model(dct_unchanged, dct, bow, dir_min_freq, dir_max_freq, file_name, processed_folder,
                       words_to_get, name_of_class,  classes, class_names, auroc, score_metric,
                       mcm, dev_percent, ppmi, kfold_hpam_dict, data_type, rewrite_all, matched_ids, model_type,
doc_topic_prior=None, topic_word_prior=None, n_topics=None,
                       top_scoring_freq=None):

    # Convert to hyper-parameter method, where hyper-parameters are:
    ##### dir_min_freq, dir_max freq
    ##### Scoring filters, aka top 200, top 400, etc, score-type
    dct_len_start = len(dct_unchanged.dfs.keys())
    if bow.shape[0] != len(dct.dfs.keys()):
        print("bow", bow.shape[0], "dct", len(dct.dfs.keys()))
        raise ValueError("Size of vocab and dict do not match")

    doc_amt = split.get_doc_amt(data_type)

    if top_scoring_freq is not None:
        wl_save = SaveLoad(rewrite=rewrite_all)
        if dir_min_freq != -1 and dir_max_freq != -1:
            no_below = int(doc_amt * dir_min_freq)
            no_above = int(doc_amt * dir_max_freq)
            print("Limit normally")
            dir = LimitWords(file_name, wl_save, dct, bow, processed_folder + "directions/words/", words_to_get, no_below,
                             no_above)
        else:
            # called this for laziness sake
            no_below = top_scoring_freq
            no_above = 0
            print("Limit numeric")
            dir = LimitWordsNumeric(file_name, wl_save, dct, bow, processed_folder + "directions/words/", words_to_get,
                                    no_below)
        dir.process_and_save()
        words_to_get = dir.getBowWordDct()
        new_word_dict = dir.getNewWordDict()

        fil_save = SaveLoad(rewrite=rewrite_all)
        fil_words = filter_bow.Filter(bow, words_to_get, new_word_dict, file_name, fil_save,
                                      processed_folder + "topic/fil/")
        fil_words.process_and_save()
        new_bow = fil_words.getNewBow()
        words = fil_words.getWords()
        new_bow_fn = fil_words.new_bow.file_name
        print("(For directions) Filtering all words that do not appear in", no_below, "documents")
    else:
        no_below = 0
        no_above = 0
        new_bow = np.asarray(bow.todense())
        words = words_to_get
        new_bow_fn = "original full bow"


    if len(new_bow) != doc_amt:
        new_bow = new_bow.transpose()
        if len(new_bow) != doc_amt:
            raise ValueError("Bow is wrong shape")

    if len(words) != len(new_bow[0]):
        raise ValueError("words dont match")

    print(len(new_bow), len(new_bow[0]))

    dir_fn = file_name + "_" + str(no_below) + "_" + str(no_above) + "_" + str(topic_word_prior) + "_" + str(doc_topic_prior) + "_" + str(n_topics)

    top_save = SaveLoad(rewrite=rewrite_all)
    topic_model = topicmodel.TopicModel(new_bow, words, doc_topic_prior, topic_word_prior, n_topics, dir_fn, "topic/model/", top_save)
    topic_model.process_and_save()
    topic_rep = topic_model.getRep()

    if len(topic_rep) != doc_amt:
        raise ValueError("topic rep wrong size")

    split_ids = split.get_split_ids(data_type, matched_ids)
    x_train, y_train, x_test, y_test, x_dev, y_dev = split.split_data(topic_rep,
                                                                      classes, split_ids,
                                                                     dev_percent_of_train=dev_percent,
                                                                      data_type=data_type)


    hpam_save = SaveLoad(rewrite=True)
    hyper_param = KFoldHyperParameter.HParam(class_names, kfold_hpam_dict, model_type, dir_fn,
                                             processed_folder + "topic/", hpam_save,
                                             False, rewrite_model=True, x_train=x_train, y_train=y_train,
                                             x_test=x_test,
                                             y_test=y_test, x_dev=x_dev, y_dev=y_dev, score_metric=score_metric,
                                             auroc=auroc, mcm=mcm, dim_names=words)
    hyper_param.process_and_save()

    top_params = hyper_param.getTopScoringParams()
    top_row_data = hyper_param.getTopScoringRowData()
    return top_params, top_row_data, new_bow_fn


def main(data_type, raw_folder, processed_folder, proj_folder="", grams=0, dir_min_freq=0.001,
         dir_max_freq=0.95, dev_percent=0.2, score_metric="avg_f1",  multiclass="OVR", LR=False,
         bonus_fn="", model_type=None, max_depth=None,
         rewrite_all=False, hp_top_freq=None, hp_top_dir=None):
    pipeline_fn = "num_stw"
    if data_type == "newsgroups":
        name_of_class = ["Newsgroups"]
    elif data_type == "sentiment":
        name_of_class = ["Sentiment"]
    elif data_type == "movies":
        name_of_class = ["Genres", "Keywords", "Ratings"]
    elif data_type == "placetypes":
        name_of_class = ["Foursquare", "Geonames", "OpenCYC"]
    elif data_type == "reuters":
        name_of_class = ["Reuters"]

    # Run a pipeline that retains numbers and removes stopwords

    dims = [200,100,50]
    window_size = [5, 10, 15]
    min_count = [1, 5, 10]
    train_epoch = [50, 100, 200]

    dims = [200,100,50]
    balance_params = ["balanced", None]
    C_params = [1.0, 0.01, 0.001, 0.0001]
    gamma_params = [1.0, 0.01, 0.001, 0.0001]

    n_estimators = [ 1000, 2000]
    max_features = [None, 'auto', 'log2']
    criterion = ["gini", "entropy"]
    max_depth = [max_depth]
    bootstrap = [True, False]
    min_samples_leaf = [1]
    min_samples_split = [2]


    doc_topic_prior = [0.001, 0.01, 0.1]
    topic_word_prior = [0.1, 0.01, 0.001]
    n_topics = [2000, 1000, 500, 200, 100, 50]
    if model_type == "GaussianSVM" or model_type == "GaussianSVMMultiClass":
        kfold_hpam_dict = {"C":C_params,
                    "class_weight":balance_params,
                           "gamma":gamma_params}
    elif model_type == "LinearSVM":
        kfold_hpam_dict = {"C":C_params,
                    "class_weight":balance_params}
    elif model_type == "RandomForest":
        kfold_hpam_dict = {"n_estimators":n_estimators,
                           "max_features":max_features,
                    "class_weight":balance_params,
                           "criterion":criterion,
                           "max_depth":max_depth,
                           "bootstrap":bootstrap,
                           "min_samples_leaf":min_samples_leaf,
                           "min_samples_split":min_samples_split}
    elif model_type[:12] == "DecisionTree":
        kfold_hpam_dict = {"max_features":max_features,
                    "class_weight":balance_params,
                           "criterion":criterion,
                           "max_depth":max_depth}

    hpam_dict = {"window_size": window_size,
                 "min_count": min_count,
                 "train_epoch": train_epoch}

    pipeline_hpam_dict = {"top_scoring_freq": hp_top_freq,"doc_topic_prior": doc_topic_prior,
                       "topic_word_prior": topic_word_prior,
                       "n_topics": n_topics }

    multi_class_method = None
    if multiclass == "MOP":
        multi_class_method = MultiOutputClassifier
    elif multiclass == "OVR":
        multi_class_method = OneVsRestClassifier
    elif multiclass == "OVO":
        multi_class_method = OneVsOneClassifier
    elif multiclass == "OCC":
        multi_class_method = OutputCodeClassifier

    for ci in range(len(name_of_class)):
        tsrds = []
        classes_save = SaveLoad(rewrite=False)
        # These were the parameters used for the previous experiments
        no_below = 0.0001
        no_above = 0.95
        bowmin = 2
        classes_freq_cutoff = 100
        # The True here and below is to remove stop words
        classes_process = util.classify.ProcessClasses(None, None, pipeline_fn, processed_folder, bowmin, no_below,
                                                       no_above, classes_freq_cutoff, True, classes_save,
                                                       name_of_class[ci])

        class_names = classes_process.getClassNames()

        corp_save = SaveLoad(rewrite=False)
        p_corpus = process_corpus.Corpus(None, None, name_of_class[ci], pipeline_fn, processed_folder,
                                         bowmin,
                                         no_below, no_above, True, corp_save)
        bow = p_corpus.getBow()
        word_list = p_corpus.getAllWords()
        classes = p_corpus.getClasses()

        for i in range(len(dims)):

            corp_save = SaveLoad(rewrite=False)
            p_corpus = process_corpus.Corpus(None, classes, name_of_class[ci], pipeline_fn, processed_folder,
                                             bowmin,
                                             no_below, no_above, True, corp_save)
            # Reload the dict because gensim is persistent otherwise
            dct = p_corpus.getBowDct()
            dct_unchanged = p_corpus.getBowDct()

            if data_type == "movies" or data_type == "placetypes":
                classifier_fn = pipeline_fn + "_" + name_of_class[i] + "_" + multiclass
                tsrd = pipeline(pipeline_fn, bow, dct, classes, class_names, word_list, processed_folder,
                                dims, kfold_hpam_dict, hpam_dict,
                                model_type=model_type, dev_percent=dev_percent, rewrite_all=rewrite_all,
                                remove_stop_words=True,
                                score_metric=score_metric, auroc=False, dir_min_freq=dir_min_freq,
                                dir_max_freq=dir_max_freq, name_of_class=name_of_class[ci],
                                classifier_fn=classifier_fn,
                                mcm=multi_class_method, ppmi=None, dct_unchanged=dct_unchanged,
                                pipeline_hpam_dict=pipeline_hpam_dict, space_name=None,
                                data_type=data_type)
            else:
                classifier_fn = pipeline_fn + "_" + multiclass
                tsrd = pipeline(pipeline_fn, bow, dct, classes, class_names, word_list, processed_folder,
                                dims, kfold_hpam_dict, hpam_dict,
                                model_type=model_type, dev_percent=dev_percent, rewrite_all=rewrite_all,
                                remove_stop_words=True,
                                score_metric=score_metric, auroc=False, dir_min_freq=dir_min_freq,
                                dir_max_freq=dir_max_freq, name_of_class=name_of_class[ci],
                                classifier_fn=classifier_fn,
                                mcm=multi_class_method, ppmi=None, dct_unchanged=dct_unchanged,
                                pipeline_hpam_dict=pipeline_hpam_dict, space_name=None,
                                data_type=data_type)
            tsrds.append(tsrd)
        # Make the combined CSV of all the dims of all the space types
        all_r = np.asarray(tsrds).transpose()
        rows = all_r[1]
        cols = np.asarray(rows.tolist()).transpose()
        col_names = all_r[0][0]
        key = all_r[2]
        dt.write_csv(processed_folder + "rank/score/csv_final/" + pipeline_fn + "reps" + model_type + "_" + name_of_class[
            ci] + ".csv", col_names, cols, key)
        print("a")


if __name__ == '__main__':
    classifiers = ["DecisionTree1","DecisionTree3", "DecisionTree2"]
    data_types = ["movies"]
    doLR = False
    dminf = -1
    dmanf = -1

    mcm = "OVR"
    bonus_fn = ""
    rewrite_all = False
    for j in range(len(data_types)):
        if data_types[j] == "placetypes":
            hp_top_freq = [20000]
        elif data_types[j] == "reuters":
            hp_top_freq = [20000]
        elif data_types[j] == "sentiment":
            hp_top_freq = [20000]
        elif data_types[j] == "newsgroups":
            hp_top_freq = [20000]
        elif data_types[j] == "movies":
            hp_top_freq = [20000]
        for i in range(len(classifiers)):
            if "1" in classifiers[i]:
                max_depths = 1
            elif "2" in classifiers[i]:
                max_depths = 2
            elif "3" in classifiers[i]:
                max_depths = 3
            else:
                max_depths = None
            main(data_types[j], "../../data/raw/"+data_types[j]+"/",  "../../data/processed/"+data_types[j]+"/", proj_folder="../../data/proj/"+data_types[j]+"/",
                                    grams=0, model_type=classifiers[i], dir_min_freq=dminf, dir_max_freq=dmanf, dev_percent=0.2,
                                    score_metric="avg_f1", max_depth=max_depths, multiclass=mcm, LR=doLR, bonus_fn=bonus_fn, rewrite_all=rewrite_all, hp_top_freq=hp_top_freq)