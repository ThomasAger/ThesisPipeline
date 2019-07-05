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
from project.bag_of_ppmi import PAV
from project.finetuning_network import FineTuneNetwork

def pipeline(file_name, classes, class_names, processed_folder, kfold_hpam_dict,
             model_type="", dev_percent=0.2, rewrite_all=False,
             score_metric="", auroc=False, name_of_class="", mcm=None,
             pipeline_hpam_dict=None,  data_type="", space=None,
             bow=None,dct=None, ranking_names=None, rank_fn=None, dir_fn=None):
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

    hyper_param = KFoldHyperParameter.RecHParam(None, classes, class_names, pipeline_hpam_dict, kfold_hpam_dict, "ft",
                                                model_type,
                                                file_name + "_ft", None, processed_folder + "ft/", hpam_save,
                                                probability=False,
                                                rewrite_model=rewrite_all, dev_percent=dev_percent,
                                                data_type=data_type, score_metric=score_metric, auroc=auroc,
                                                matched_ids=matched_ids, end_fn_added=name_of_class + "_ftc",
                                                mcm=mcm,
                                                hpam_params=[file_name, processed_folder,  rewrite_all, data_type, space, class_names,
                      kfold_hpam_dict, model_type, name_of_class, score_metric, mcm, classes, dev_percent, matched_ids,
                                                             rank_fn, dir_fn, ranking_names, bow, dct])
    hyper_param.process_and_save()
    print("END OF SPACE")
    return hyper_param.getTopScoringRowData()

def get_dp(space, direction):
    ranking = np.empty(len(space))
    for i in range(len(space)):
        ranking[i] = np.dot(direction, space[i])
    return ranking

def ft_pipeline( file_name, processed_folder,  rewrite_all, data_type, space, class_names,
                      kfold_hpam_dict, model_type, name_of_class, score_metric, multi_class_method, classes, dev_percent, matched_ids,
                 rank_fn, dir_fn, ranking_names_fn, bow, dct, hidden_layer_size, epoch, activation_function, use_hidden):

    if use_hidden is False:
        hidden_layer_size = 0
        activation_function = "None"

    ranking = dt.import2dArray(rank_fn)

    if ".npy" in ranking_names_fn:
        ranking_names = np.load(ranking_names_fn)
    else:
        ranking_names = dt.import1dArray(ranking_names_fn)
    doc_amt = split.get_doc_amt(data_type)

    dir = np.load(dir_fn)

    if len(dir) <= 200 and len(dir[0]) <= 200:
        use_cluster = True


    if len(ranking[0]) != doc_amt and len(ranking) != doc_amt:
        raise ValueError("Rank len does not equal doc amt")

    if len(ranking) == doc_amt:
        ranking = ranking.transpose()

    if len(space) != doc_amt:
        raise ValueError("Space len does not equal doc amt")

    if use_cluster is False:
        dir = dir.transpose()
        if False in (np.isclose(ranking.transpose()[0], get_dp(space, dir[0])))  :
            raise ValueError("Rankings or space do not match")
    else:
        if False in (np.isclose(ranking[0], get_dp(space, dir[0])))  :
            raise ValueError("Rankings or space do not match")



    pav_save = SaveLoad(rewrite=rewrite_all)
    pav = PAV(file_name, processed_folder + "boc/", dct.token2id, bow, ranking, ranking_names, False, pav_save)
    pav.process_and_save()
    boc = pav.getPAV()

    ft_fn = file_name + "_"+str(hidden_layer_size) + "_"+str(epoch)+"_" + str(use_hidden) + "_" + str(activation_function) + "_"
    ft_save = SaveLoad(rewrite=rewrite_all)
    ft = FineTuneNetwork(file_name, processed_folder + "ft/", space, dir, ranking, boc, "", hidden_layer_size, activation_function, epoch, ft_save, use_hidden)
    ft.process_and_save()
    rankings = ft.getRanks()
    ranking_fn = ft.output_ranks.file_name

    if data_type == "placetypes" or data_type == "movies":
        ft_fn = ft_fn + "_" + name_of_class
    split_ids = split.get_split_ids(data_type, matched_ids)
    x_train, y_train, x_test, y_test, x_dev, y_dev = split.split_data(rankings,
                                                                      classes, split_ids,
                                                                      dev_percent_of_train=dev_percent,
                                                                      data_type=data_type)
    if ft_fn == "" or ft_fn is None:
        raise ValueError("ftfn is  nothing")
    hpam_save = SaveLoad(rewrite=rewrite_all)
    hyper_param = KFoldHyperParameter.HParam(class_names, kfold_hpam_dict, model_type, ft_fn,
                                             processed_folder + "ft/", hpam_save,
                                             False, rewrite_model=rewrite_all, x_train=x_train, y_train=y_train,
                                             x_test=x_test,
                                             y_test=y_test, x_dev=x_dev, y_dev=y_dev, score_metric=score_metric,
                                             auroc=False, mcm=multi_class_method)

    hyper_param.process_and_save()
    # Get the scores for those rankings
    return hyper_param.getTopScoringParams(), hyper_param.getTopScoringRowData(), ranking_fn


def main(data_type, raw_folder, processed_folder, proj_folder="", grams=0, model_type="LinearSVM", dir_min_freq=0.001,
         dir_max_freq=0.95, dev_percent=0.2, score_metric="avg_f1", max_depth=None, multiclass="OVR", LR=False,
         bonus_fn="", rewrite_all=False, clusters=False):
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
    spaces = []
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
    if clusters is False:
        for j in range(len(name_of_class)):
            csv = dt.read_csv(csv_fn + name_of_class[j] + ".csv")
            rank_fn = csv.sort_values("avg_f1", ascending=False).values[0][rank_id]
            print(rank_fn)
            rank_fn_array.append(rank_fn)

            word_fn = "_".join(rank_fn.split("_")[:-1])+ "_words.npy"
            print(word_fn)
            word_fn_array.append(word_fn)

            dir_fn = csv.sort_values("avg_f1", ascending=False).values[0][dir_id]
            dir_fn_array.append(dir_fn)
            space_name = rank_fn.split("/")[-1:][0][:-4]
            space_name_array.append(space_name)
    else:
        for j in range(len(name_of_class)):
            if data_type == "newsgroups":
                if model_type == "DecisionTree3":
                    csv_fn = processed_folder + "clusters/score/csv_final/" + "num_stw_num_stw_200_MDS_ndcg_2000_5000_0_rankreps" + model_type + "_"+ name_of_class[j]
                elif model_type == "DecisionTree2":
                    csv_fn = processed_folder + "clusters/score/csv_final/" + "num_stw_num_stw_200_MDS_ndcg_2000_5000_0_rankreps" + model_type + "_"+ name_of_class[j]
                elif model_type == "DecisionTree1":
                    csv_fn = processed_folder + "clusters/score/csv_final/" + "num_stw_num_stw_200_MDS_ndcg_2000_5000_0_rankreps" + model_type + "_"+ name_of_class[j]
            elif data_type == "placetypes":
                if model_type == "DecisionTree3":
                    if name_of_class[j] == "Foursquare":
                        csv_fn = processed_folder + "clusters/score/csv_final/" + "num_stw_num_stw_50_PCA_kappa_1000_10000_0_rankreps" + model_type + "_"+ name_of_class[j]
                    elif  name_of_class[j] == "OpenCYC":
                        csv_fn = processed_folder + "clusters/score/csv_final/" + "num_stw_num_stw_50_AWVEmp_ndcg_2000_10000_0_rankreps" + model_type + "_"+ name_of_class[j]
                    elif  name_of_class[j] == "Geonames":
                        csv_fn = processed_folder + "clusters/score/csv_final/" + "num_stw_num_stw_100_AWVEmp_ndcg_1000_20000_0_rankreps" + model_type + "_"+ name_of_class[j]
                elif model_type == "DecisionTree2":
                    if name_of_class[j] == "Foursquare":
                        csv_fn = processed_folder + "clusters/score/csv_final/" + "num_stw_num_stw_50_AWVEmp_kappa_2000_20000_0_rankreps" + model_type + "_"+ name_of_class[j]
                    elif  name_of_class[j] == "OpenCYC":
                        csv_fn = processed_folder + "clusters/score/csv_final/" + "num_stw_num_stw_100_AWVEmp_ndcg_2000_20000_0_rankreps" + model_type + "_"+ name_of_class[j]
                    elif  name_of_class[j] == "Geonames":
                        csv_fn = processed_folder + "clusters/score/csv_final/" + "num_stw_num_stw_50_AWVEmp_ndcg_2000_10000_0_rankreps" + model_type + "_"+ name_of_class[j]
                elif model_type == "DecisionTree1":
                    if name_of_class[j] == "Foursquare":
                        csv_fn = processed_folder + "clusters/score/csv_final/" + "num_stw_num_stw_200_AWVEmp_ndcg_1000_20000_0_rankreps" + model_type + "_"+ name_of_class[j]
                    elif  name_of_class[j] == "OpenCYC":
                        csv_fn = processed_folder + "clusters/score/csv_final/" + "num_stw_num_stw_100_AWVEmp_ndcg_1000_5000_0_rankreps" + model_type + "_"+ name_of_class[j]
                    elif  name_of_class[j] == "Geonames":
                        csv_fn = processed_folder + "clusters/score/csv_final/" + "num_stw_num_stw_50_AWVEmp_ndcg_1000_10000_0_rankreps" + model_type + "_"+ name_of_class[j]
            elif data_type == "sentiment":
                if model_type == "DecisionTree3":
                    csv_fn = processed_folder + "clusters/score/csv_final/" + "num_stw_num_stw_50_PCA_ndcg_2000_20000_0_rankreps" + model_type + "_"+ name_of_class[j]
                elif model_type == "DecisionTree2":
                    csv_fn = processed_folder + "clusters/score/csv_final/" + "num_stw_num_stw_100_D2V_ndcg_1000_10000_0_rankreps" + model_type + "_"+ name_of_class[j]
                elif model_type == "DecisionTree1":
                    csv_fn = processed_folder + "clusters/score/csv_final/" + "num_stw_num_stw_100_D2V_ndcg_2000_20000_0_rankreps" + model_type + "_"+ name_of_class[j]
            elif data_type == "reuters":
                if model_type == "DecisionTree3":
                    csv_fn = processed_folder + "clusters/score/csv_final/" + "num_stw_num_stw_200_MDS_ndcg_2000_5000_0_rankreps" + model_type + "_" + name_of_class[j]+ "_" + "False"
                elif model_type == "DecisionTree2":
                    csv_fn = processed_folder + "clusters/score/csv_final/" + "num_stw_num_stw_200_MDS_ndcg_2000_5000_0_rankreps" + model_type + "_"+ name_of_class[j]+ "_" + "False"
                elif model_type == "DecisionTree1":
                    csv_fn = processed_folder + "clusters/score/csv_final/" + "num_stw_num_stw_200_MDS_ndcg_2000_5000_0_rankreps" + model_type + "_"+ name_of_class[j] + "_"+ "False"
            elif data_type == "movies":
                if model_type == "DecisionTree3":
                    if name_of_class[j] == "Genres":
                        csv_fn = processed_folder + "clusters/score/csv_final/" + "num_stw_num_stw_200_MDS_ndcg_1000_10000_0_rankreps" + model_type + "_"
                    elif  name_of_class[j] == "Keywords":
                        csv_fn = processed_folder + "clusters/score/csv_final/" + "num_stw_num_stw_50_MDS_ndcg_2000_20000_0_rankreps" + model_type + "_"
                    elif  name_of_class[j] == "Ratings":
                        csv_fn = processed_folder + "clusters/score/csv_final/" + "num_stw_num_stw_100_MDS_ndcg_1000_20000_0_rankreps" + model_type + "_"
                elif model_type == "DecisionTree2":
                    if name_of_class[j] == "Genres":
                        csv_fn = processed_folder + "clusters/score/csv_final/" + "num_stw_num_stw_100_MDS_ndcg_1000_20000_0_rankreps" + model_type + "_"
                    elif  name_of_class[j] == "Keywords":
                        csv_fn = processed_folder + "clusters/score/csv_final/" + "num_stw_num_stw_100_MDS_ndcg_2000_5000_0_rankreps" + model_type + "_"
                    elif  name_of_class[j] == "Ratings":
                        csv_fn = processed_folder + "clusters/score/csv_final/" + "num_stw_num_stw_50_PCA_ndcg_2000_10000_0_rankreps" + model_type + "_"
                elif model_type == "DecisionTree1":
                    if name_of_class[j] == "Genres":
                        csv_fn = processed_folder + "clusters/score/csv_final/" + "num_stw_num_stw_200_MDS_ndcg_2000_10000_0_rankreps" + model_type + "_"
                    elif  name_of_class[j] == "Keywords":
                        csv_fn = processed_folder + "clusters/score/csv_final/" + "num_stw_num_stw_100_MDS_ndcg_2000_20000_0_rankreps" + model_type + "_"
                    elif  name_of_class[j] == "Ratings":
                        csv_fn = processed_folder + "clusters/score/csv_final/" + "num_stw_num_stw_100_PCA_ndcg_2000_10000_0_rankreps" + model_type + "_"
            csv = dt.read_csv(csv_fn + ".csv")
            rank_fn = csv.sort_values("avg_f1", ascending=False).values[0][rank_id]
            print(rank_fn)
            rank_fn_array.append(rank_fn)

            split_fn = rank_fn.split("/")
            split_fn[6] = "names"
            split_fn = "/".join(split_fn)
            word_fn = "_".join(split_fn.split("_")[:-3]) + ".txt"
            print(word_fn)
            word_fn_array.append(word_fn)

            dir_fn = csv.sort_values("avg_f1", ascending=False).values[0][dir_id]
            dir_fn_array.append(dir_fn)
            space_name = rank_fn.split("/")[-1:][0][:-4]
            space_name_array.append(space_name)

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

            dim = int(rank_fns[i][j].split("/")[-1:][0].split("_")[4])
            type = rank_fns[i][j].split("/")[-1:][0].split("_")[5]
            if type == "MDS":
                if data_type != "sentiment":
                    mds_identifier = "_" + str(dim) + "_MDS"
                    mds_fn = pipeline_fn + mds_identifier
                    import_fn = processed_folder + "rep/mds/" + mds_fn + ".npy"
                    mds_space = dt.import2dArray(import_fn)
                    if len(mds_space) < len(mds_space[0]):
                        mds_space = mds_space.transpose()
                    space = mds_space
                    #space_names[i].append(mds_fn)
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
                #space_names[i].append(awv_fn)
                space = awv_space
            elif type == "PCA":
                pca_identifier = "_" + str(dim) + "_PCA"
                pca_fn = pipeline_fn + pca_identifier
                pca_instance = pca.PCA(None, None, dim,
                                       pca_fn, processed_folder + "rep/pca/", SaveLoad(rewrite=False))
                pca_instance.process_and_save()
                pca_space = pca_instance.getRep()
                #space_names[i].append(pca_fn)
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
                    #space_names[i].append(doc2vec_fn)
                    space = d2v_space

            epoch = [50, 300]
            hidden_layer_size = [1]
            activation_function = ["linear",  "tanh"]
            use_hidden = [True]
            pipeline_hpam_dict = {"epoch": epoch,
                                  "hidden_layer_size": hidden_layer_size,
                                  "activation_function": activation_function,
                                  "use_hidden": use_hidden}

            tsrd = pipeline(space_names[i][j], classes, class_names, processed_folder, kfold_hpam_dict,
                            model_type=model_type, dev_percent=dev_percent, rewrite_all=rewrite_all, score_metric=score_metric,
                            auroc=False, name_of_class=name_of_class[j], mcm=multi_class_method, pipeline_hpam_dict=pipeline_hpam_dict,
                            data_type=data_type, space=space, bow=bow, dct=dct, rank_fn=rank_fns[i][j], ranking_names=word_fns[i][j],
                            dir_fn=dir_fns[i][j])

            # Make the combined CSV of all the dims of all the space types
            all_r = tsrd
            cols = all_r[1]
            col_names = all_r[0]
            key = all_r[2]
            dt.write_csv(
                processed_folder + "ft/score/csv_final/" + space_names[i][j] + "reps" + model_type + "_" +
                name_of_class[j] + "_" + ".csv",
                col_names, cols, key)
            print("a")




def init():
    classifiers = ["DecisionTree1", "DecisionTree2", "DecisionTree3"]
    data_type = [ "reuters"]
    use_clusters = [True, False]
    for j in range(len(data_type)):
        doLR = False
        dminf = -1
        dmanf = -1

        multi_class_method = "OVR"
        bonus_fn = ""
        rewrite_all = False
        print("iterating through classifiers")
        for u_clusters in use_clusters:
            for i in range(len(classifiers)):
                if "1" in classifiers[i]:
                    max_depths = 1
                elif "2" in classifiers[i]:
                    max_depths = 2
                elif "3" in classifiers[i]:
                    max_depths = 3
                else:
                    max_depths = None
                print(classifiers[i])
                main(data_type[j], "../../data/raw/" + data_type[j] + "/", "../../data/processed/" + data_type[j] + "/",
                     proj_folder="../../data/proj/" + data_type[j] + "/",
                     grams=0, model_type=classifiers[i], dir_min_freq=dminf, dir_max_freq=dmanf, dev_percent=0.2,
                     score_metric="avg_f1", max_depth=max_depths, multiclass=multi_class_method, LR=doLR, bonus_fn=bonus_fn,
                     rewrite_all=rewrite_all, clusters=u_clusters)

if __name__ == '__main__':
    print("starting")
    init()