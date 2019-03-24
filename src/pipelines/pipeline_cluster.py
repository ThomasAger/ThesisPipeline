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
from util import py

import KFoldHyperParameter

# The overarching pipeline to obtain all prerequisite data for the derrac pipeline
# Todo: Better standaradize the saving/loading
last_dct = []


def pipeline(file_name, top_ranks, bow, dct, classes, class_names, words_to_get, processed_folder, dims, kfold_hpam_dict,
             hpam_dict,
             model_type="", dev_percent=0.2, rewrite_all=False, remove_stop_words=True,
             score_metric="", auroc=False, dir_min_freq=0.001, dir_max_freq=0.95, name_of_class="", space_name="",
             classifier_fn="", mcm=None, top_scoring_dirs=2000, score_type="kappa", ppmi=None, dct_unchanged=None,
             pipeline_hpam_dict=None):
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
    hyper_param = KFoldHyperParameter.RecHParam(None, classes, class_names, pipeline_hpam_dict, kfold_hpam_dict, "cluster",
                                                model_type,
                                                file_name, None, processed_folder + "rank/", hpam_save,
                                                probability=False,
                                                rewrite_model=rewrite_all, dev_percent=dev_percent,
                                                data_type=data_type, score_metric=score_metric, auroc=auroc,
                                                matched_ids=matched_ids, end_fn_added=name_of_class,
                                                mcm=mcm,
                                                hpam_params=[dct_unchanged, dct, bow, dir_min_freq, dir_max_freq,
                                                             file_name, processed_folder,
                                                             words_to_get, top_ranks, name_of_class, model_type, classes,
                                                             class_names, auroc, score_metric,
                                                             mcm, dev_percent, ppmi, kfold_hpam_dict])
    hyper_param.process_and_save()
    print("END OF SPACE")
    return hyper_param.getTopScoringRowData()


def cluster_pipeline(dct_unchanged, dct, bow, dir_min_freq, dir_max_freq, file_name, processed_folder,
                       words_to_get, top_ranks, name_of_class, model_type, classes, class_names, auroc, score_metric,
                       mcm, dev_percent, ppmi, kfold_hpam_dict, top_scoring_dir=None, top_scoring_freq=None, matched_ids=None):

    dct_len_start = len(dct_unchanged.dfs.keys())
    if bow.shape[0] != len(dct.dfs.keys()) or len(dct.dfs.keys()) != ppmi.shape[0]:
        print("bow", bow.shape[0], "dct", len(dct.dfs.keys()), "ppmi", ppmi.shape[0])
        raise ValueError("Size of vocab and dict do not match")

    doc_amt = split.get_doc_amt(data_type)

    # Normalize the directions (so that euclidian distance is equal to cosine similarity)

    # Get the clusters For the cluster input parameters with the directions as input

    # Get the rankings on the clusters



    # Find which one is best with the model type

    split_ids = split.get_split_ids(data_type, matched_ids)
    x_train, y_train, x_test, y_test, x_dev, y_dev = split.split_data(fil_rank,
                                                                      classes, split_ids,
                                                                      dev_percent_of_train=dev_percent,
                                                                      data_type=data_type)

    hyper_param = KFoldHyperParameter.HParam(class_names, kfold_hpam_dict, model_type, dir_fn,
                                             processed_folder + "rank/", hpam_save,
                                             False, rewrite_model=rewrite_this, x_train=x_train, y_train=y_train,
                                             x_test=x_test,
                                             y_test=y_test, x_dev=x_dev, y_dev=y_dev, score_metric=score_metric,
                                             auroc=auroc, mcm=mcm, dim_names=words)
    hyper_param.process_and_save()

    tsp.append(hyper_param.getTopScoringParams())
    tsrd.append(hyper_param.getTopScoringRowData())
    rfn.append(gtr.rank.file_name)
    f1s.append(hyper_param.getTopScoringRowData()[1][1])  # F1 Score for the current scoring metric

    best_ind = np.flipud(np.argsort(f1s))[0]
    top_params = tsp[best_ind]
    top_row_data = tsrd[best_ind]
    top_rank = rfn[best_ind]

    all_r = np.asarray(tsrd).transpose()
    rows = all_r[1]
    cols = np.asarray(rows.tolist()).transpose()
    col_names = all_r[0][0]
    key = all_r[2]
    dt.write_csv(processed_folder + "rank/score/csv_averages/" + file_name + "_" + str(no_above) + "_" + str(no_below)
                 + "reps" + model_type + "_" + name_of_class + ".csv", col_names, cols, key)
    print("Pipeline completed, saved as, " + processed_folder + "rank/score/csv_averages/" + file_name + "_" + str(
        no_above) + "_" + str(no_below)
          + "reps" + model_type + "_" + name_of_class + ".csv")
    return top_params, top_row_data, top_rank


def main(data_type, raw_folder, processed_folder, proj_folder="", grams=0, model_type="LinearSVM", dir_min_freq=0.001,
         dir_max_freq=0.95, dev_percent=0.2, score_metric="avg_f1", max_depth=None, multiclass="OVR", LR=False,
         bonus_fn="",
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

    window_size = [5, 10, 15]
    min_count = [1, 5, 10]
    train_epoch = [50, 100, 200]

    dims = [200, 100, 50]
    balance_params = ["balanced", None]
    C_params = [1.0, 0.01, 0.001, 0.0001]
    gamma_params = [1.0, 0.01, 0.001, 0.0001]

    n_estimators = [1000, 2000]
    max_features = [None, 'auto', 'log2']
    criterion = ["gini", "entropy"]
    max_depth = [max_depth]
    bootstrap = [True, False]
    min_samples_leaf = [1]
    min_samples_split = [2]

    # Run a pipeline that retains numbers and removes stopwords

    if model_type == "GaussianSVM" or model_type == "GaussianSVMMultiClass":
        kfold_hpam_dict = {"C": C_params,
                           "class_weight": balance_params,
                           "gamma": gamma_params}
    elif model_type == "LinearSVM":
        kfold_hpam_dict = {"C": C_params,
                           "class_weight": balance_params}
    elif model_type == "RandomForest":
        kfold_hpam_dict = {"n_estimators": n_estimators,
                           "max_features": max_features,
                           "class_weight": balance_params,
                           "criterion": criterion,
                           "max_depth": max_depth,
                           "bootstrap": bootstrap,
                           "min_samples_leaf": min_samples_leaf,
                           "min_samples_split": min_samples_split}
    elif model_type[:12] == "DecisionTree":
        kfold_hpam_dict = {"max_features": max_features,
                           "class_weight": balance_params,
                           "criterion": criterion,
                           "max_depth": max_depth}
    hpam_dict = {"window_size": window_size,
                 "min_count": min_count,
                 "train_epoch": train_epoch}

    pipeline_hpam_dict = {"top_freq": hp_top_freq, "top_dir": hp_top_dir}

    multi_class_method = None
    if multiclass == "MOP":
        multi_class_method = MultiOutputClassifier
    elif multiclass == "OVR":
        multi_class_method = OneVsRestClassifier
    elif multiclass == "OVO":
        multi_class_method = OneVsOneClassifier
    elif multiclass == "OCC":
        multi_class_method = OutputCodeClassifier

    tsrds = []

    rank_fns = []

    placetypes_fn = processed_folder + "rank/score/csv_final/" + "num_stw_num_stw_50_PCArepsDecisionTree3_"
    reuters_fn = ""
    newsgroups_fn = ""
    movies_fn = ""
    sentiment_fn = ""

    all_fns = [placetypes_fn]#, reuters_fn, newsgroups_fn, movies_fn, sentiment_fn]
    space_names = []
    for i in range(len(all_fns)):
        if i == 0: # Placetypes
            rank_fn_array = []
            space_name_array = []
            for j in range(len(name_of_class)):
                csv = dt.read_csv(all_fns[i] + name_of_class[j] + ".csv")
                rank_fn = csv.sort_values("avg_f1", ascending=False).values[0][5]
                rank_fn_array.append(rank_fn)
                space_name = rank_fn.split("/")[-1:][0][:-4]
                space_name_array.append(space_name)
            rank_fns.append(rank_fn_array)
            space_names.append(space_name_array)


    classes_save = SaveLoad(rewrite=False)
    # These were the parameters used for the previous experiments
    no_below = 0.0001
    no_above = 0.95
    bowmin = 2
    classes_freq_cutoff = 100
    # The True here and below is to remove stop words

    for i in range(len(rank_fns)):
        for j in range(len(name_of_class)):
            classes_process = util.classify.ProcessClasses(None, None, pipeline_fn, processed_folder, bowmin, no_below,
                                                           no_above, classes_freq_cutoff, True, classes_save,
                                                           name_of_class[j])

            class_names = classes_process.getClassNames()

            corp_save = SaveLoad(rewrite=False)
            p_corpus = process_corpus.Corpus(None, None, name_of_class[j], pipeline_fn, processed_folder,
                                             bowmin,
                                             no_below, no_above, True, corp_save)
            classes = p_corpus.getClasses()
            bow = p_corpus.getBow()
            word_list = p_corpus.getAllWords()

            final_fn = pipeline_fn

            dct = p_corpus.getBowDct()
            dct_unchanged = p_corpus.getBowDct()
            classifier_fn = final_fn + "_" + name_of_class[i] + "_" + multiclass
            if data_type == "movies" or data_type == "placetypes":
                tsrd = pipeline(final_fn, rank_fns[i][j], bow, dct, classes, class_names, word_list, processed_folder, dims,
                         kfold_hpam_dict, hpam_dict,
                         model_type=model_type, dev_percent=dev_percent, rewrite_all=rewrite_all,
                         remove_stop_words=True,
                         score_metric=score_metric, auroc=False, dir_min_freq=dir_min_freq,
                         dir_max_freq=dir_max_freq, name_of_class=name_of_class[j], classifier_fn=classifier_fn,
                         mcm=multi_class_method,  dct_unchanged=dct_unchanged,
                         pipeline_hpam_dict=pipeline_hpam_dict, space_name=space_names[i][j])
            """
            tsrds.append(tsrd)
            # Make the combined CSV of all the dims of all the space types
            all_r = np.asarray(tsrds).transpose()
            rows = all_r[1]
            cols = np.asarray(rows.tolist()).transpose()
            col_names = all_r[0][0]
            key = all_r[2]
            dt.write_csv(
                processed_folder + "rank/score/csv_final/" + final_fn + "reps" + model_type + "_" + name_of_class[j] + ".csv",
                col_names, cols, key)
            print("a")
            """


max_depths = [3]
classifiers = ["DecisionTree3"]
data_type = "placetypes"
doLR = False
dminf = -1
dmanf = -1

if data_type == "placetypes":
    hp_top_freq = [5000, 10000, 20000]
    hp_top_dir = [1000, 2000]
elif data_type == "reuters":
    hp_top_freq = [5000, 10000, 20000]
    hp_top_dir = [1000, 2000]
elif data_type == "sentiment":
    hp_top_freq = [5000, 10000, 20000]
    hp_top_dir = [1000, 2000]
elif data_type == "newsgroups":
    hp_top_freq = [5000, 10000, 20000]
    hp_top_dir = [1000, 2000]
elif data_type == "movies":
    hp_top_freq = [5000, 10000, 20000]
    hp_top_dir = [1000, 2000]

multi_class_method = "OVR"
bonus_fn = ""
rewrite_all = False
if __name__ == '__main__':
    for i in range(len(classifiers)):
        main(data_type, "../../data/raw/" + data_type + "/", "../../data/processed/" + data_type + "/",
             proj_folder="../../data/proj/" + data_type + "/",
             grams=0, model_type=classifiers[i], dir_min_freq=dminf, dir_max_freq=dmanf, dev_percent=0.2,
             score_metric="avg_f1", max_depth=max_depths[i], multiclass=multi_class_method, LR=doLR, bonus_fn=bonus_fn,
             rewrite_all=rewrite_all,
             hp_top_dir=hp_top_dir, hp_top_freq=hp_top_freq)