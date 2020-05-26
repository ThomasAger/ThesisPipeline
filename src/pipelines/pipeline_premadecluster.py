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
from project.get_directions import GetDirections, GetDirectionsSimple
from score.classify import MultiClassScore
from project.get_rankings import GetRankings, GetRankingsStreamed, GetRankingsNoSave
from project.get_ndcg import GetNDCG, GetNDCGStreamed
from rep import pca, ppmi, awv
from project.get_tsr import GetTopScoringRanks, GetTopScoringRanksStreamed
from util import py
from util.normalize import NormalizeZeroMean
import KFoldHyperParameter
from model.kmeans import KMeansCluster
from model.derrac_cluster import DerracCluster
from util.consolidate_classes import ConsolidateClasses
import scipy.sparse as sp

# The overarching pipeline to obtain all prerequisite data for the derrac pipeline
# Todo: Better standaradize the saving/loading
last_dct = []


def pipeline(file_name, classes, class_names, processed_folder, kfold_hpam_dict,
             model_type="", dev_percent=0.2, rewrite_all=False,
             score_metric="", auroc=False, name_of_class="",
             pipeline_hpam_dict=None, cluster_amt=0, data_type="",  cluster_method=None,
             svm_clusters=False, multi_class_method=None, rankings=None, cluster_names=None, rank_fn=None):

    try:
        matched_ids = np.load(processed_folder + "classes/" + name_of_class + "_matched_ids.npy")
    except FileNotFoundError:
        try:
            matched_ids = []
            class_entities = dt.import1dArray(processed_folder + "classes/" + name_of_class + "_entities.txt")
            print(processed_folder + "classes/" + name_of_class + "_entities.txt")
            entity_names = dt.import1dArray(processed_folder + "corpus/entity_names.txt")
            print(processed_folder + "corpus/entity_names.txt")
            for i in range(len(class_entities)):
                for j in range(len(entity_names)):
                    if class_entities[i] == entity_names[j]:
                        matched_ids.append(j)
                        break
            np.save(processed_folder + "classes/" + name_of_class + "_matched_ids.npy", matched_ids)
        except FileNotFoundError:
            matched_ids = None

    hpam_save = SaveLoad(rewrite=rewrite_all)

    # Folds and space are determined inside of the method for this hyper-parameter selection, as it is stacked
    print(file_name)
    print(name_of_class)
    print(cluster_method)
    print(multi_class_method)
    top_params, top_row_data, rank_fn, cluster_names, dir_fn, rank_fn = cluster_pipeline(file_name, processed_folder, rewrite_all,  data_type,  class_names,
                      kfold_hpam_dict, model_type, name_of_class, score_metric, multi_class_method, classes, dev_percent,
                      matched_ids, rankings, cluster_names, rank_fn)


    print("END OF SPACE")
    return top_row_data


def cluster_pipeline( file_name, processed_folder, rewrite_all,  data_type,  class_names,
                      kfold_hpam_dict, model_type, name_of_class, score_metric, multi_class_method, classes, dev_percent,
                      matched_ids, rankings, cluster_names, rank_fn):

    if data_type == "placetypes" or data_type == "movies":
        dir_fn = file_name + "_" + name_of_class
    split_ids = split.get_split_ids(data_type, matched_ids)
    x_train, y_train, x_test, y_test, x_dev, y_dev = split.split_data(rankings,
                                                                      classes, split_ids,
                                                                      dev_percent_of_train=dev_percent,
                                                                      data_type=data_type)

    if dir_fn == "" or dir_fn is None:
        raise ValueError("Dir_fn is  nothing")
    hpam_save = SaveLoad(rewrite=rewrite_all)
    print(multi_class_method)

    hyper_param = KFoldHyperParameter.HParam(class_names, kfold_hpam_dict, model_type, dir_fn,
                                         processed_folder + "clusters/", hpam_save,
                                         False, rewrite_model=rewrite_all, x_train=x_train, y_train=y_train,
                                         x_test=x_test,
                                         y_test=y_test, x_dev=x_dev, y_dev=y_dev, score_metric=score_metric,
                                         auroc=False, mcm=multi_class_method, dim_names=cluster_names,
                                         feature_names=cluster_names)
    hyper_param.process_and_save()


    # Get the scores for those rankings
    return hyper_param.getTopScoringParams(), hyper_param.getTopScoringRowData(), rank_fn,  cluster_names, dir_fn, rank_fn


def main(data_type, raw_folder, processed_folder, proj_folder="", grams=0, model_type="LinearSVM", dir_min_freq=0.001,
         dir_max_freq=0.95, dev_percent=0.2, score_metric="avg_f1", max_depth=None, multiclass="OVR", LR=False,
         bonus_fn="", cluster_amt=None, cluster_methods=None,
         rewrite_all=None, top_dir_amt=None, svm_clusters=False, use_space=None, use_space_name=None, use_dir_fn=None, use_dir_names=None,
         use_cluster_fn=None, use_cluster_name_fn=None, rank_fn=None):

    orig_top_dir_amt = top_dir_amt
    pipeline_fn = "num_stw"
    name_of_class = None
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

    n_estimators = [ 1000, 2000]
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

    multi_class_method = None
    if multiclass == "MOP":
        multi_class_method = MultiOutputClassifier
    elif multiclass == "OVR":
        multi_class_method = OneVsRestClassifier
    elif multiclass == "OVO":
        multi_class_method = OneVsOneClassifier
    elif multiclass == "OCC":
        multi_class_method = OutputCodeClassifier

    rank_fns = []
    dir_fns = []
    word_fns = []
    feature_fns = []
    if data_type == "placetypes" or data_type == "movies":
        csv_fn = processed_folder + "rank/score/csv_final/" + "num_stw_num_stw_50_PCAreps"+model_type+"_"
    elif data_type == "reuters" or data_type == "newsgroups" or data_type == "sentiment":
        csv_fn = processed_folder + "rank/score/csv_final/" + "num_stw_num_stw_50_D2Vreps"+model_type+"_"

    space_names = []
    # Sometimes directions is last in the csv, otherwise rank is

    classes_save = SaveLoad(rewrite=False)

    # These were the parameters used for the previous experiments
    no_below = 0.0001
    no_above = 0.95
    bowmin = 2
    classes_freq_cutoff = 100


    fn_clusters = ["ae0", "ae1", "ae2", "ae3", "ae4", "ae04", "aeall"]
    cluster_amt = [200,    200,    100,   50,    25, 400, 575]
    # The True here and below is to remove stop words
    print(rank_fns)
    for j in range(len(name_of_class)):

        tsrds = []
        for i in range(len(fn_clusters)):
            rankings = np.load(use_cluster_fn[i])
            cluster_names = np.load(use_cluster_name_fn[i])

            classes_process = util.classify.ProcessClasses(None, None, pipeline_fn, processed_folder, bowmin, no_below,
                                                           no_above, classes_freq_cutoff, True, classes_save,
                                                           name_of_class[j])

            class_names = classes_process.getClassNames()

            corp_save = SaveLoad(rewrite=False)
            p_corpus = process_corpus.Corpus(None, None, name_of_class[j], pipeline_fn, processed_folder,
                                             bowmin,
                                             no_below, no_above, True, corp_save)
            classes = p_corpus.getClasses()
            tsrd = pipeline(fn_clusters[i],   classes, class_names, processed_folder, kfold_hpam_dict,
                            model_type=model_type, dev_percent=dev_percent, rewrite_all=rewrite_all, score_metric=score_metric,
                            auroc=False, name_of_class=name_of_class[j], multi_class_method=multi_class_method,
                            cluster_amt=cluster_amt[i], data_type=data_type,
                            rankings=rankings, cluster_names=cluster_names, rank_fn=rank_fn[i])

            tsrds.append(tsrd)

        # Make the combined CSV of all the dims of all the space types
        all_r = np.asarray(tsrds).transpose()
        rows = all_r[1]
        cols = np.asarray(rows.tolist()).transpose()
        col_names = all_r[0][0]
        key = all_r[2]
        dt.write_csv(
            processed_folder + "clusters/score/csv_final/" + "reps" +  model_type + "_"
           + "_" + str(svm_clusters) + "_"+ str(fn_clusters[i]) + "_" + name_of_class[j] + ".csv",
            col_names, cols, key)
        print("a")



def init():
    classifiers = ["DecisionTree3", "DecisionTree2", "DecisionTree1"]
    data_type = ["movies"]

    c_fn_start = "../../data/raw\Data_NeSy16\Input Vectors/"
    c_name_fn_start = "../../data/raw\Data_NeSy16\Cluster Classes/"
    use_cluster_fn = [c_fn_start + "L0_nodupe.npy", c_fn_start + "L1_nodupe.npy", c_fn_start + "L2_nodupe.npy",
                      c_fn_start + "L3_nodupe.npy", c_fn_start + "L4_nodupe.npy", c_fn_start + "L0_and_4.npy"
                      , c_fn_start + "LAllConcat.npy"]
    use_cluster_name_fn = [c_name_fn_start + "L0 Cluster.npy", c_name_fn_start + "L1 Cluster.npy", c_name_fn_start + "L2 Cluster.npy",
                           c_name_fn_start + "L3 Cluster.npy", c_name_fn_start + "L4 Cluster.npy", c_name_fn_start + "L0_and_4.npy"
                      , c_name_fn_start + "LAllConcat.npy"]

    for j in range(len(data_type)):
        doLR = False
        dminf = -1
        dmanf = -1
        cluster_amt = [50, 100, 200]
        if data_type[j] == "placetypes":
            cluster_amt = [50, 100, 200]
            top_dir_amt = [2, 1.0, 4, 6, 8]
        elif data_type[j] == "reuters":
            cluster_amt = [50, 100, 200]
            top_dir_amt = [2, 1.0, 4, 6, 8]
        elif data_type[j] == "sentiment":
            cluster_amt = [50, 100, 200]
            top_dir_amt = [2, 1.0, 4, 6, 8]
        elif data_type[j] == "newsgroups":
            cluster_amt = [50, 100, 200]
            top_dir_amt = [2, 1.0, 4, 6, 8]
        elif data_type[j] == "movies":
            cluster_amt = [50, 100, 200]
            top_dir_amt = [2, 1.0, 4, 6, 8]

        cluster_methods = ["kmeans"]

        svm_clusters = [False]

        multiclass = "OVR"
        bonus_fn = ""
        rewrite_all = False#"2019 11 22 14 32"
        print("iterating through classifiers")
        for i in range(len(classifiers)):
            if "1" in classifiers[i]:
                max_depths = 1
            elif "2" in classifiers[i]:
                max_depths = 2
            elif "3" in classifiers[i]:
                max_depths = 3
            else:
                max_depths = None
            for k in range(len(svm_clusters)):
                print(classifiers[i])
                main(data_type[j], "../../data/raw/" + data_type[j] + "/", "../../data/processed/" + data_type[j] + "/",
                     proj_folder="../../data/proj/" + data_type[j] + "/",
                     grams=0, model_type=classifiers[i], dir_min_freq=dminf, dir_max_freq=dmanf, dev_percent=0.2,
                     score_metric="avg_f1", max_depth=max_depths, multiclass=multiclass, LR=doLR, bonus_fn=bonus_fn,
                     rewrite_all=rewrite_all, cluster_amt=cluster_amt, cluster_methods=cluster_methods, top_dir_amt=top_dir_amt,
                     svm_clusters=svm_clusters[k],  use_cluster_fn=use_cluster_fn,
                     use_cluster_name_fn=use_cluster_name_fn, rank_fn=use_cluster_fn)

if __name__ == '__main__':
    print("starting")
    init()