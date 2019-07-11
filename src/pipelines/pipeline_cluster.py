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


def pipeline(file_name, top_dirs_fn,classes, class_names, processed_folder, kfold_hpam_dict,
             model_type="", dev_percent=0.2, rewrite_all=False,
             score_metric="", auroc=False, name_of_class="", mcm=None,
             pipeline_hpam_dict=None, cluster_amt=0, data_type="", dir_names=None, space=None, cluster_method=None, n_init=10, max_iter=300, tol=1e-4, init="k-means++",
             svm_clusters=False,bow=None,dct=None):
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

    hpam_save = SaveLoad(rewrite=rewrite_all)

    # Folds and space are determined inside of the method for this hyper-parameter selection, as it is stacked
    print(file_name)

    hyper_param = KFoldHyperParameter.RecHParam(None, classes, class_names, pipeline_hpam_dict, kfold_hpam_dict, "cluster",
                                                model_type,
                                                file_name + "_"+ str(cluster_amt), None, processed_folder + "clusters/", hpam_save,
                                                probability=False,
                                                rewrite_model=rewrite_all, dev_percent=dev_percent,
                                                data_type=data_type, score_metric=score_metric, auroc=auroc,
                                                matched_ids=matched_ids, end_fn_added=name_of_class + "_" + str(svm_clusters) + "_" + cluster_method,
                                                mcm=mcm,
                                                hpam_params=[file_name, processed_folder, cluster_amt, rewrite_all, top_dirs_fn, data_type, dir_names, space,class_names,
                      kfold_hpam_dict, model_type, name_of_class, score_metric, mcm, classes, dev_percent, matched_ids, cluster_method, svm_clusters,bow, dct])
    hyper_param.process_and_save()
    print("END OF SPACE")
    return hyper_param.getTopScoringRowData()


def cluster_pipeline( file_name, processed_folder, cluster_amt, rewrite_all, top_dir_fn, data_type, word_fn, space, class_names,
                      kfold_hpam_dict, model_type, name_of_class, score_metric, multi_class_method, classes, dev_percent, matched_ids, cluster_method,
                      svm_clusters, bow, dct, n_init=10, max_iter=300, tol=1e-4, init="k-means++", top_dir_amt = 2):


    doc_amt = split.get_doc_amt(data_type)

    if len(space) != doc_amt:
        raise ValueError("Space len does not equal doc amt")

    top_dirs = np.load(top_dir_fn).transpose()
    top_words = np.load(word_fn)

    cluster_save = SaveLoad(rewrite=rewrite_all)
    if cluster_method == "kmeans":
        dir_fn = file_name + "_" + str(n_init) + "_" + str(max_iter) + "_" + str(tol) + "_" + str(init) + "_" + str(cluster_amt) + "_" + cluster_method
        # Normalize the directions (so that euclidian distance is equal to cosine similarity)
        norm_save = SaveLoad(rewrite=rewrite_all)
        normalize = NormalizeZeroMean(top_dirs, file_name, processed_folder + "directions/norm/", norm_save)
        normalize.process_and_save()
        top_dirs = normalize.getNormalized()
        cluster = KMeansCluster( top_dirs, cluster_amt, dir_fn, processed_folder + "clusters/", cluster_save, top_words)
    elif cluster_method == "derrac":
        dir_fn = file_name + "_" + str(top_dir_amt) + "_" + str(cluster_amt) + "_" + cluster_method
        cluster = DerracCluster(top_dirs, cluster_amt, dir_fn, processed_folder + "clusters/", cluster_save, top_words, top_dir_amt)
    else:
        raise ValueError("No cluster method found")

    # Get the clusters For the cluster input parameters with the directions as input
    cluster.process_and_save()
    cluster_dir = cluster.getCentroids()
    cluster_dir_name = cluster.getDirName()
    cluster_names = np.asarray(cluster.getClusterNames())

    if svm_clusters is True:
        dir_fn = dir_fn + "_svmCFix"

        cluster_dirs = cluster.getClusters()
        token2id = dct.token2id
        new_word_dict = {}
        for i in range(len(cluster_names)):
            new_word_dict[cluster_names[i]] = i
        cc_save = SaveLoad(rewrite=rewrite_all)
        cc = ConsolidateClasses(token2id, bow, cluster_dirs, cluster_names, dir_fn, processed_folder + "clusters/bow/", cc_save )
        cc.process_and_save()
        cluster_bows = cc.getCentroids()
        dir_save = SaveLoad(rewrite=rewrite_all)
        dir = GetDirectionsSimple(cluster_bows, space, dir_save, dir_fn,processed_folder + "clusters/directions/", LR=False)
        dir.process_and_save()
        cluster_dir = dir.getDirections()
        # Make the new bow





    cluster_dict = {}
    for i in range(len(cluster_names)):
        cluster_dict[cluster_names[i]] = i
    # Get the rankings on the clusters
    rank_save = SaveLoad(rewrite=rewrite_all)
    rank = GetRankingsNoSave(cluster_dir, space, cluster_dict, rank_save, dir_fn, processed_folder + "clusters/", "best", cluster_amt)
    rank.process_and_save()
    rankings = rank.getRankings()
    rankings = rankings.transpose()
    if len(rankings) != doc_amt:
        raise ValueError("Rankings len does not equal doc amt")

    if data_type == "placetypes" or data_type == "movies":
        dir_fn = dir_fn + "_" + name_of_class
    split_ids = split.get_split_ids(data_type, matched_ids)
    x_train, y_train, x_test, y_test, x_dev, y_dev = split.split_data(rankings,
                                                                      classes, split_ids,
                                                                      dev_percent_of_train=dev_percent,
                                                                      data_type=data_type)
    if dir_fn == "" or dir_fn is None:
        raise ValueError("Dir_fn is  nothing")
    hpam_save = SaveLoad(rewrite=rewrite_all)
    hyper_param = KFoldHyperParameter.HParam(class_names, kfold_hpam_dict, model_type, dir_fn,
                                             processed_folder + "clusters/", hpam_save,
                                             False, rewrite_model=rewrite_all, x_train=x_train, y_train=y_train,
                                             x_test=x_test,
                                             y_test=y_test, x_dev=x_dev, y_dev=y_dev, score_metric=score_metric,
                                             auroc=False, mcm=multi_class_method, dim_names=cluster_names,
                                             feature_names=cluster_names)

    hyper_param.process_and_save()
    # Get the scores for those rankings
    return hyper_param.getTopScoringParams(), hyper_param.getTopScoringRowData(), rank.rankings.file_name,  cluster_names, dir_fn, cluster_dir_name


def main(data_type, raw_folder, processed_folder, proj_folder="", grams=0, model_type="LinearSVM", dir_min_freq=0.001,
         dir_max_freq=0.95, dev_percent=0.2, score_metric="avg_f1", max_depth=None, multiclass="OVR", LR=False,
         bonus_fn="", cluster_amt=None, cluster_methods=None,
         rewrite_all=False, top_dir_amt=None, svm_clusters=False):
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
    if data_type == "movies":
        rank_id = 6
        dir_id = 5
    else:
        rank_id = 5
        dir_id = 6

    rank_fn_array = []
    dir_fn_array = []
    word_fn_array = []
    space_name_array = []
    for j in range(len(name_of_class)):
        csv = dt.read_csv(csv_fn + name_of_class[j] + ".csv")
        space_name_internal = []
        rank_fn_internal =[]
        word_fn_internal = []
        dir_fn_internal = []
        for z in range(len(csv.values)):
            rank_fn = csv.values[z][rank_id]
            print(rank_fn)
            rank_fn_internal.append(rank_fn)

            word_fn = "_".join(rank_fn.split("_")[:-1])+ "_words.npy"
            print(word_fn)
            word_fn_internal.append(word_fn)

            dir_fn = csv.values[z][dir_id]
            dir_fn_internal.append(dir_fn)

            space_name = rank_fn.split("/")[-1:][0][:-4]
            space_name_internal.append(space_name)
        rank_fn_array.append(rank_fn_internal)
        word_fn_array.append(word_fn_internal)
        space_name_array.append(space_name_internal)
        dir_fn_array.append(dir_fn_internal)

    rank_fns.append(rank_fn_array)
    dir_fns.append(dir_fn_array)
    word_fns.append(word_fn_array)
    space_names.append(space_name_array)

    classes_save = SaveLoad(rewrite=False)

    # These were the parameters used for the previous experiments
    no_below = 0.0001
    no_above = 0.95
    bowmin = 2
    classes_freq_cutoff = 100
    # The True here and below is to remove stop words
    print(rank_fns)
    for i in range(len(rank_fns)):
        for j in range(len(name_of_class)):
            for a in range(len(space_names[i][j])):
                print(j)
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
                dct = p_corpus.getBowDct()
                space = None

                dim = int(rank_fns[i][j][a].split("/")[-1:][0].split("_")[4])
                type = rank_fns[i][j][a].split("/")[-1:][0].split("_")[5]
                if type == "MDS":
                    if data_type != "sentiment":
                        mds_identifier = "_" + str(dim) + "_MDS"
                        mds_fn = pipeline_fn + mds_identifier
                        import_fn = processed_folder + "rep/mds/" + mds_fn + ".npy"
                        mds_space = dt.import2dArray(import_fn)
                        space = mds_space
                        """
                        metadata_fn = processed_folder + "bow/metadata/" + "num_stw_remove.npy"
        
                        if len(mds_space) != 18302:
                            del_ids = np.load(metadata_fn)
                            mds_space = np.delete(mds_space, del_ids, axis=0)
                            np.save(import_fn,mds_space)
                        """

                elif type == "AWVEmp":
                    awv_identifier = "_" + str(dim) + "_AWVEmp"
                    awv_fn = pipeline_fn + awv_identifier
                    awv_instance = awv.AWV(None, dim, awv_fn, processed_folder + "rep/awv/", SaveLoad(rewrite=False))
                    awv_instance.process_and_save()
                    awv_space = awv_instance.getRep()
                    space = awv_space
                elif type == "PCA":
                    pca_identifier = "_" + str(dim) + "_PCA"
                    pca_fn = pipeline_fn + pca_identifier
                    pca_instance = pca.PCA(None, None, dim,
                                           pca_fn, processed_folder + "rep/pca/", SaveLoad(rewrite=False))
                    pca_instance.process_and_save()
                    pca_space = pca_instance.getRep()
                    #space_names.append(pca_fn)
                    space = pca_space
                elif type == "D2V":
                    if data_type != "movies" and data_type != "placetypes":
                        doc2vec_identifier = "_" + str(dim) + "_D2V"
                        doc2vec_fn = pipeline_fn + doc2vec_identifier
                        classifier_fn = pipeline_fn + "_" + name_of_class[j] + "_"

                        wv_path_d2v = os.path.abspath("../../data/raw/glove/" + "glove.6B.300d.txt")

                        corpus_fn = processed_folder + "corpus/" + "num_stw_corpus_processed.txt"
                        hpam_dict["dim"] = [dim]
                        hpam_dict["corpus_fn"] = [corpus_fn]
                        hpam_dict["wv_path"] = [wv_path_d2v]

                        # Have to leave classes in due to messy method
                        hyper_param = KFoldHyperParameter.RecHParam(None, classes, None, hpam_dict, hpam_dict, "d2v",
                                                                    "LinearSVM",
                                                                    doc2vec_fn, classifier_fn, processed_folder + "rep/",
                                                                    SaveLoad(rewrite=False), data_type=data_type,
                                                                    score_metric=score_metric)
                        d2v_space, __unused = hyper_param.getTopScoringSpace()
                        #space_names.append(doc2vec_fn)
                        space = d2v_space
                tsrds = []
                for c in range(len(cluster_amt)):
                    for cm in range(len(cluster_methods)):
                        if cluster_methods[cm] == "kmeans":
                            n_init = [5, 10, 50]
                            max_iter = [100, 300, 1000]
                            tol = [0.001, 0.0001, 0.00001, 0.0]
                            top_dir_amt = [0]
                            pipeline_hpam_dict = {"n_init": n_init,
                                                  "max_iter": max_iter,
                                                  "tol": tol,
                                                  "top_dir_amt": top_dir_amt}
                        elif cluster_methods[cm] == "derrac":
                            n_init = [0]
                            max_iter = [0]
                            tol = [0]
                            top_dir_amt = orig_top_dir_amt

                            pipeline_hpam_dict = {"n_init": n_init,
                                                  "max_iter": max_iter,
                                                  "tol": tol,
                                                  "top_dir_amt": top_dir_amt}
                        print(cluster_amt[c])
                        tsrd = pipeline(space_names[i][j][a],  dir_fns[i][j][a],  classes, class_names,processed_folder, kfold_hpam_dict,
                                        model_type=model_type, dev_percent=dev_percent, rewrite_all=rewrite_all, score_metric=score_metric,
                                        auroc=False, name_of_class=name_of_class[j], mcm=multi_class_method, pipeline_hpam_dict=pipeline_hpam_dict,
                                        cluster_amt=cluster_amt[c], data_type=data_type, dir_names=word_fns[i][j][a], space=space, cluster_method=cluster_methods[cm],
                                        svm_clusters=svm_clusters, bow=bow, dct=dct)
                        tsrds.append(tsrd)
                # Make the combined CSV of all the dims of all the space types
                all_r = np.asarray(tsrds).transpose()
                rows = all_r[1]
                cols = np.asarray(rows.tolist()).transpose()
                col_names = all_r[0][0]
                key = all_r[2]
                dt.write_csv(processed_folder + "clusters/score/csv_final/" + space_names[i][j][a] +  str(a) + "reps" + model_type + "_"
                             + name_of_class[j] + "_" + str(svm_clusters)  + ".csv",
                    col_names, cols, key)
                print("a")


def init():
    classifiers = ["DecisionTree1","DecisionTree2","DecisionTree3"]
    data_type = ["placetypes"]
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

        cluster_methods = ["derrac", "kmeans"]

        svm_clusters = [False]

        multi_class_method = "OVR"
        bonus_fn = ""
        rewrite_all = False
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
                     score_metric="avg_f1", max_depth=max_depths, multiclass=multi_class_method, LR=doLR, bonus_fn=bonus_fn,
                     rewrite_all=rewrite_all, cluster_amt=cluster_amt, cluster_methods=cluster_methods, top_dir_amt=top_dir_amt,
                     svm_clusters=svm_clusters[k])

if __name__ == '__main__':
    print("starting")
    init()