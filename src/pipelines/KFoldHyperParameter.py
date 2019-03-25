
from sklearn.model_selection import KFold
from model.svm import GaussianSVM
from model.svm import LinearSVM
from common.Method import Method
from itertools import product
from util.save_load import SaveLoad
from score import classify
import numpy as np
import os
from common.SaveLoadPOPO import SaveLoadPOPO
from util import check_util
from rep import d2v
from model.randomforest import RandomForest
from model.decisiontree import DecisionTree
from util import split
from pipelines import pipeline_single_dir
from pipelines import pipeline_cluster

def get_grid_params(hpam_dict):
    hyperparam_array = list(hpam_dict.values())
    hyperparam_names = list(hpam_dict.keys())
    all_params = list(
        product(
            *hyperparam_array
        )
    )
    all_p = []
    for i in range(len(all_params)):
        pdict = {}
        for j in range(len(hyperparam_names)):
            pdict[hyperparam_names[j]] = all_params[i][j]
        all_p.append(pdict)
    return all_p

class MasterHParam(Method):
    dim_names = None
    classes = None
    dev = None
    file_name = None
    space = None
    output_folder = None
    model_type = None
    end_file_name = ""
    class_names = None
    probability = None
    averaged_csv_data = None
    file_names = None
    all_p = None
    auroc = False
    fscore = False
    acc = False
    kappa = False
    rewrite_model = False
    hpam_dict = None
    score_metric = None
    x_dev = None
    y_dev = None
    p_score_dicts = None
    # The CSV data that corresponds to the highest scoring row for the score_metric
    top_scoring_row_data = None
    final_score_on_dev = None
    top_scoring_params = None
    mcm = None

    def __init__(self, class_names=None, hpam_dict=None, model_type=None, file_name=None, output_folder=None, save_class=None, probability=None, score_metric="avg_f1", rewrite_model=False, auroc=False, fscore=True, acc=True, kappa=True, mcm=None, dim_names = None):

        self.rewrite_model = rewrite_model
        # Metric that determines what will be returned to the overall hyper-parameter method
        self.score_metric = score_metric
        self.class_names = class_names
        self.probability = probability
        self.hpam_dict = hpam_dict
        self.auroc = auroc
        self.fscore = fscore
        self.acc = acc
        self.kappa = kappa
        self.all_p = get_grid_params(hpam_dict)
        self.file_name = file_name
        self.output_folder = output_folder
        self.model_type = model_type
        self.mcm = mcm
        self.file_names = []
        self.dim_names = dim_names
        self.end_file_name = self.file_name + "_Kfold" + str(self.dev) + str(generateNumber(hpam_dict)) + self.model_type
        super().__init__(file_name, save_class)

    def trainClassifier(self, model):
        model.process_and_save()
        pred = model.getPred()
        prob = None
        if self.probability:
            prob = model.getProba()
        return pred, prob

    def process(self):
        super().process()

    def selectClassifier(self, all_p, x_train, y_train, x_test, y_test):
        svm_save = SaveLoad(rewrite=self.rewrite_model)
        model = None
        model_fn = None
        file_name = self.file_name
        if self.model_type == "LinearSVM":
            model_fn = file_name + "_Dev" + "_" + str(len(x_test)) + "_Balanced_" + str(all_p["class_weight"]) \
                       + "_C_" + str(all_p["C"]) + "_Prob_" + str(self.probability) + "_" + self.model_type
            print("Running", model_fn)
            model = LinearSVM(x_train, y_train, x_test, y_test,
                              self.output_folder + "svm/" + model_fn, svm_save, C=all_p["C"],
                              class_weight=all_p["class_weight"], probability=self.probability, verbose=False,
                              mcm=self.mcm)

        elif self.model_type == "GaussianSVM":
            param_fn = "_Balanced_" + str(all_p["class_weight"]) \
                       + "_C_" + str(all_p["C"]) + "_Gam_" + str(all_p["gamma"]) + "_Prob_" + str(
                self.probability) + "_" + self.model_type

            model_fn = file_name + "_Dev"+ "_" + str(len(x_test))  + param_fn
            print("Running", model_fn)

            model = GaussianSVM(x_train, y_train, x_test, y_test,
                                self.output_folder + "svm/" + model_fn, svm_save, gamma=all_p["gamma"],
                                C=all_p["C"],
                                class_weight=all_p["class_weight"], probability=self.probability, verbose=False,
                                mcm=self.mcm)
        elif self.model_type == "RandomForest":
            param_fn = "MClass_Balanced_" + str(all_p["class_weight"]) \
                       + "_Estim_" + str(all_p["n_estimators"]) + "_Features_" + str(
                all_p["max_features"]) + "_BS_" + str(all_p["bootstrap"]) + "_MD_" + str(
                all_p["max_depth"]) + "_MSL_" + str(all_p["min_samples_leaf"]) + "_MSS_" + str(
                all_p["min_samples_split"]) + "_" + self.model_type

            model_fn = file_name + "_Dev"+ "_" + str(len(x_test))  + param_fn
            print("Running", model_fn)
            model = RandomForest(x_train, y_train, x_test, y_test,
                                 self.output_folder + "rf/" + model_fn, svm_save,
                                 n_estimators=all_p["n_estimators"], bootstrap=all_p["bootstrap"],
                                 max_depth=all_p["max_depth"],
                                 min_samples_leaf=all_p["min_samples_leaf"],
                                 min_samples_split=all_p["min_samples_split"],
                                 class_weight=all_p["class_weight"], max_features=all_p["max_features"],
                                 probability=self.probability, verbose=False, mcm=self.mcm)
        elif self.model_type[:12] == "DecisionTree":
            param_fn = "MClass_Balanced_" + str(all_p["class_weight"]) \
                        + "_Features_" + str(
                all_p["max_features"]) + "_MD_" + str(
                all_p["max_depth"]) + "_" + self.model_type

            model_fn = file_name + "_Dev"+ "_" + str(len(x_test))  + param_fn
            print("Running", model_fn)
            model = DecisionTree(x_train, y_train, x_test, y_test,
                                 self.output_folder + "dt/" + model_fn, svm_save,
                                 max_depth=all_p["max_depth"],
                                 class_weight=all_p["class_weight"], max_features=all_p["max_features"],
                                 probability=self.probability, verbose=False, mcm=self.mcm)

        return model, model_fn

    # Using the row data prepared to make a CSV, find the scoring metric and retunr those scores
    def find_on_row_data(self, row_data):
        for i in range(len(row_data[0])):
            if row_data[0][i] == self.score_metric:
                return row_data[1][i]


# There's a way bette rto do all of this but I just pushed it through for time's sake.
class RecHParam(MasterHParam):

    folds = None
    hyperparam_names = None
    hyperparams_array = None
    average_file_names = None
    hpam_model_type = None
    final_arrays = None
    kfold_hpam_dict = None
    dev_percent = None
    data_type = None
    classify_fn = None
    matched_ids = None
    hpam_method = None
    hpam_params = None
    rank_fn = None
    fn_addition = None
    top_scoring_features = None
    dir_fn = None
    all_row_data = None

    def __init__(self, space, classes, class_names, hpam_dict, kfold_hpam_dict, hpam_model_type, model_type, file_name, classify_fn, output_folder, save_class, probability=None, rewrite_model=False, auroc=True, fscore=True, acc=True, kappa=True, dev_percent=0.2, score_metric=None, data_type=None, matched_ids=None, mcm=None, dim_names=None,
                 hpam_method=None, hpam_params=None, fn_addition=None, end_fn_added="", name_of_class=""):
        self.kfold_hpam_dict = kfold_hpam_dict
        self.hpam_model_type = hpam_model_type
        self.matched_ids = matched_ids
        self.dev_percent=dev_percent
        self.data_type = data_type
        self.space = space
        self.classes = classes
        self.classify_fn = classify_fn
        self.hpam_method = hpam_method
        self.hpam_params = hpam_params
        self.average_file_names = []
        self.all_p = get_grid_params(hpam_dict)
        self.dim_names = dim_names
        self.fn_addition = fn_addition
        self.end_file_name = file_name + "_Kfold" + str(None) + str(
            generateNumber(hpam_dict)) + model_type + end_fn_added

        super().__init__(rewrite_model=rewrite_model, auroc=auroc, fscore=fscore, acc=acc, kappa=kappa, model_type=model_type, output_folder=output_folder,
                         file_name=file_name, probability=probability, class_names=class_names,  save_class=save_class, hpam_dict=hpam_dict,
                         score_metric=score_metric, mcm=mcm, dim_names=dim_names)

    def getTopScoringRowData(self):
        if self.processed is False:
            return self.save_class.load(self.top_scoring_row_data)
        return self.top_scoring_row_data.value

    def makePopos(self):
        self.final_arrays = SaveLoadPOPO(self.final_arrays, self.output_folder + "score/csv_averages/" + self.end_file_name + ".csv", "csv")
        self.top_scoring_row_data = SaveLoadPOPO(self.top_scoring_row_data, self.output_folder + "score/csv_averages/" + self.end_file_name + "Top"+self.score_metric+".csv", "csv")
        self.top_scoring_params = SaveLoadPOPO(self.top_scoring_params, self.output_folder + "score/csv_averages/top_params/" + self.end_file_name + ".npy", "npy")
        self.top_scoring_features = SaveLoadPOPO(self.top_scoring_features, self.output_folder + "score/csv_final/top/" + self.end_file_name + ".npy", "npy")
        #self.all_row_data =  SaveLoadPOPO(self.all_row_data, self.output_folder + "score/csv_final/" + self.end_file_name + "All.csv", "csv")

    def makePopoArray(self):
        self.popo_array = [self.final_arrays, self.top_scoring_params, self.top_scoring_row_data]

    def process(self):
        col_names = []
        indexes = []
        self.rank_fn = []
        self.dir_fn = []
        averaged_csv_data = []
        self.top_scoring_params.value = []
        for i in range(len(self.all_p)):
            if self.hpam_model_type == "d2v":
                doc2vec_save = SaveLoad(rewrite=self.rewrite_model)
                identifier = "_WS_" + str(self.all_p[i]["window_size"]) + "_MC_" + \
                             str(self.all_p[i]["min_count"]) + "_TE_" + str(self.all_p[i]["train_epoch"]) + "_D_"+str(self.all_p[i]["dim"]) + "_D2V"
                doc2vec_fn = self.file_name + identifier
                d2v_classify_fn = self.classify_fn + identifier
                doc2vec_instance = d2v.D2V(self.all_p[i]["corpus_fn"], self.all_p[i]["wv_path"], doc2vec_fn,
                                           self.output_folder + "d2v/", doc2vec_save, self.all_p[i]["dim"], window_size=self.all_p[i]["window_size"],
                                           min_count=self.all_p[i]["min_count"], train_epoch=self.all_p[i]["train_epoch"]
                                           )
                doc2vec_instance.process_and_save()
                doc2vec_space = doc2vec_instance.getRep()

                split_ids = split.get_split_ids(self.data_type, self.matched_ids)
                x_train, y_train, x_test, y_test, x_dev, y_dev = split.split_data(doc2vec_space,
                                                                                  self.classes, split_ids,
                                                                                  dev_percent_of_train=self.dev_percent)
                hpam_save = SaveLoad(rewrite=self.rewrite_model)
                hyper_param = HParam(self.class_names,
                                     self.kfold_hpam_dict, self.model_type, d2v_classify_fn,
                                     self.output_folder, hpam_save, self.probability, rewrite_model=self.rewrite_model, x_train=x_train,
                                     y_train=y_train, x_test=x_test, y_test=y_test, x_dev=x_dev, y_dev=y_dev, final_score_on_dev=True, auroc=self.auroc, mcm=self.mcm)
                hyper_param.process_and_save()
                self.top_scoring_params.value.append(hyper_param.getTopScoringParams())
                top_scoring_row_data = hyper_param.getTopScoringRowData()
                averaged_csv_data.append(top_scoring_row_data[1])
                col_names = top_scoring_row_data[0]
                indexes.append(top_scoring_row_data[2][0])
            elif self.hpam_model_type == "dir":
                if self.all_p[i]["top_dir"] > self.all_p[i]["top_freq"]:
                    continue
                top_params, top_row_data, top_rank, top_dir = pipeline_single_dir.direction_pipeline(*self.hpam_params, top_scoring_freq=self.all_p[i]["top_freq"], top_scoring_dir=self.all_p[i]["top_dir"])

                self.top_scoring_params.value.append(top_params)
                top_scoring_row_data = top_row_data
                averaged_csv_data.append(top_scoring_row_data[1])
                col_names = top_scoring_row_data[0]
                indexes.append(top_scoring_row_data[2][0])
                self.rank_fn.append(top_rank)
                self.dir_fn.append(top_dir)
            elif self.hpam_model_type == "cluster":
                top_params, top_row_data, cluster_rank =  pipeline_cluster.cluster_pipeline(*self.hpam_params, n_init=self.all_p[i]["n_init"], max_iter=self.all_p[i]["max_iter"],
                                                                                            tol=self.all_p[i]["tol"])
                self.top_scoring_params.value.append(top_params)
                top_scoring_row_data = top_row_data
                averaged_csv_data.append(top_scoring_row_data[1])
                col_names = top_scoring_row_data[0]
                indexes.append(top_scoring_row_data[2][0])
                self.rank_fn.append(cluster_rank)
        self.final_arrays.value = []
        self.final_arrays.value.append(col_names)
        self.final_arrays.value.append(np.asarray(averaged_csv_data).transpose())
        self.final_arrays.value.append(indexes)
        if self.hpam_model_type == "d2v":
            self.getTopScoringByMetric()
        elif self.hpam_model_type == "dir" or self.hpam_model_type == "cluster":
            print("skipped")
            index = self.getTopScoring()
            if self.hpam_model_type == "cluster":
                space = np.load(self.rank_fn[index]).transpose()
                self.getTopScoringCluster(space, index)
            else:
                space = np.load(self.rank_fn[index])
                self.getTopScoringByMetricDir(space, index)

        super().process()
    def getTopScoring(self):
        if self.final_arrays.value is None or len(self.final_arrays.value) == 0:
            self.final_arrays.value = self.save_class.load(self.final_arrays)
            self.top_scoring_params.value = self.save_class.load(self.top_scoring_params)
        list_of_scores = self.find_on_row_data(self.final_arrays.value)
        index_sorted = np.flipud(np.argsort(list_of_scores))[0]
        return index_sorted

    def getTopScoringByMetricDir(self, space, index):
        split_ids = split.get_split_ids(self.data_type, self.matched_ids)
        x_train, y_train, x_test, y_test, x_dev, y_dev = split.split_data(space,
                                                                          self.classes, split_ids,
                                                                          dev_percent_of_train=self.dev_percent)

        model, model_fn = self.selectClassifier(self.top_scoring_params.value[index], x_train, y_train, x_test, y_test)
        model_pred, __unused = self.trainClassifier(model)
        score_save = SaveLoad(rewrite=self.rewrite_model, load_all=True)
        score = classify.selectScore(y_test, model_pred, None, file_name=model_fn,
                                         output_folder=self.output_folder + "score/", save_class=score_save,
                                         verbose=True, class_names=self.class_names,
                                         fscore=self.fscore, acc=self.acc, kappa=self.kappa, auroc=self.auroc)
        score.process_and_save()

        score_dict = score.get()
        # This order cannot be changed as this is the order it is imported as.
        col_names = ["avg_acc", "avg_f1", "avg_kappa", "avg_prec", "avg_recall", "rank_fn", "dir_fn"]
        avg_array = [score_dict[col_names[0]], score_dict[col_names[1]],
                     score_dict[col_names[2]], score_dict[col_names[3]],
                     score_dict[col_names[4]], self.rank_fn[index], self.dir_fn[index]]
        self.top_scoring_features = space
        self.top_scoring_row_data.value = [np.asarray(col_names), np.asarray(avg_array), np.asarray([model_fn])]

    def getTopScoringCluster(self, space, index):
        split_ids = split.get_split_ids(self.data_type, self.matched_ids)
        x_train, y_train, x_test, y_test, x_dev, y_dev = split.split_data(space,
                                                                          self.classes, split_ids,
                                                                          dev_percent_of_train=self.dev_percent)

        model, model_fn = self.selectClassifier(self.top_scoring_params.value[index], x_train, y_train, x_test, y_test)
        model_pred, __unused = self.trainClassifier(model)
        score_save = SaveLoad(rewrite=self.rewrite_model, load_all=True)
        score = classify.selectScore(y_test, model_pred, None, file_name=model_fn,
                                     output_folder=self.output_folder + "score/", save_class=score_save,
                                     verbose=True, class_names=self.class_names,
                                     fscore=self.fscore, acc=self.acc, kappa=self.kappa, auroc=self.auroc)
        score.process_and_save()

        score_dict = score.get()
        # This order cannot be changed as this is the order it is imported as.
        col_names = ["avg_acc", "avg_f1", "avg_kappa", "avg_prec", "avg_recall", "rank_fn"]
        avg_array = [score_dict[col_names[0]], score_dict[col_names[1]],
                     score_dict[col_names[2]], score_dict[col_names[3]],
                     score_dict[col_names[4]], self.rank_fn[index]]
        self.top_scoring_features = space
        self.top_scoring_row_data.value = [np.asarray(col_names), np.asarray(avg_array), np.asarray([model_fn])]


    #If you need other metrics than F1 just do metric = "acc" then index is 0 etc.
    def getTopScoringByMetric(self):
        self.getTopScoringByMetricDir(*self.getTopScoringSpace())
    """ Potentially repeated code so removed
    def getTopScoringByScore(self, space, index):
        split_ids = split.get_split_ids(self.data_type, self.matched_ids)
        x_train, y_train, x_test, y_test, x_dev, y_dev = split.split_data(space,
                                                                          self.classes, split_ids,
                                                                          dev_percent_of_train=self.dev_percent)
        model, model_fn = self.selectClassifier(self.top_scoring_params.value[index], x_train, y_train, x_test, y_test)

        model_pred, __unused = self.trainClassifier(model)
        score_save = SaveLoad(rewrite=self.rewrite_model, load_all=True)
        score = classify.selectScore(y_test, model_pred, None, file_name=model_fn,
                                         output_folder=self.output_folder + "score/", save_class=score_save,
                                         verbose=True, class_names=self.class_names,
                                         fscore=self.fscore, acc=self.acc, kappa=self.kappa, auroc=self.auroc)
        score.process_and_save()

        score_dict = score.get()
        # This order cannot be changed as this is the order it is imported as.
        col_names = ["avg_acc", "avg_f1", "avg_kappa", "avg_prec", "avg_recall"]
        avg_array = [score_dict[col_names[0]], score_dict[col_names[1]],
                     score_dict[col_names[2]], score_dict[col_names[3]],
                     score_dict[col_names[4]]]
        self.top_scoring_row_data.value = [np.asarray(col_names), np.asarray(avg_array), np.asarray([model_fn])]
"""
    def getTopScoringSpace(self):
        index_sorted = self.getTopScoring()
        print("Training best parameters on test data not dev data")
        doc2vec_save = SaveLoad(rewrite=False)
        doc2vec_fn = self.file_name + "_WS_" + str(self.all_p[index_sorted]["window_size"]) + "_MC_" + str(
            self.all_p[index_sorted]["min_count"]) + "_TE_" + str(self.all_p[index_sorted]["train_epoch"]) + "_D_" + str(
            self.all_p[index_sorted]["dim"]) + "_D2V"

        doc2vec_instance = d2v.D2V(self.all_p[index_sorted]["corpus_fn"], self.all_p[index_sorted]["wv_path"], doc2vec_fn,
                                   self.output_folder + "d2v/", doc2vec_save, self.all_p[index_sorted]["dim"],
                                   window_size=self.all_p[index_sorted]["window_size"],
                                   min_count=self.all_p[index_sorted]["min_count"], train_epoch=self.all_p[index_sorted]["train_epoch"]
                                   )

        doc2vec_instance.process_and_save()
        doc2vec_space = doc2vec_instance.getRep()
        return doc2vec_space, index_sorted



# There's a way bette rto do all of this but I just pushed it through for time's sake.
class DirectionsHParam(MasterHParam):

    folds = None
    hyperparam_names = None
    hyperparams_array = None
    average_file_names = None
    hpam_model_type = None
    final_arrays = None
    kfold_hpam_dict = None
    dev_percent = None
    data_type = None
    classify_fn = None
    matched_ids = None
    all_row_data = None

    def __init__(self, space, classes, class_names, hpam_dict, kfold_hpam_dict, hpam_model_type, model_type, file_name, classify_fn, output_folder, save_class, probability=None, rewrite_model=False, auroc=True, fscore=True, acc=True, kappa=True, dev_percent=0.2, score_metric=None, data_type=None, matched_ids=None):
        self.kfold_hpam_dict = kfold_hpam_dict
        self.hpam_model_type = hpam_model_type
        self.matched_ids = matched_ids
        self.dev_percent=dev_percent
        self.data_type = data_type
        self.space = space
        self.classes = classes
        self.classify_fn = classify_fn
        self.average_file_names = []
        self.all_p = get_grid_params(hpam_dict)

        self.end_file_name = file_name + "_Kfold" + str(None) + str(
            generateNumber(hpam_dict)) + model_type

        super().__init__(rewrite_model=rewrite_model, auroc=auroc, fscore=fscore, acc=acc, kappa=kappa, model_type=model_type, output_folder=output_folder,
                         file_name=file_name, probability=probability, class_names=class_names,  save_class=save_class, hpam_dict=hpam_dict,
                         score_metric=score_metric)

    def getTopScoringRowData(self):
        return self.save_class.load(self.top_scoring_row_data)

    def makePopos(self):
        self.final_arrays = SaveLoadPOPO(self.final_arrays, self.output_folder + "score/csv_averages/" + self.end_file_name + ".csv", "csv")
        self.top_scoring_row_data = SaveLoadPOPO(self.top_scoring_row_data, self.output_folder + "score/csv_averages/" + self.end_file_name + "Top"+self.score_metric+".csv", "csv")
        self.top_scoring_params = SaveLoadPOPO(self.top_scoring_params, self.output_folder + "score/csv_averages/top_params/" + self.end_file_name + ".npy", "npy")
        self.all_row_data =  SaveLoadPOPO(self.all_row_data, self.output_folder + "score/csv_averages/" + self.end_file_name + "All.csv", "csv")

    def makePopoArray(self):
        self.popo_array = [self.final_arrays, self.top_scoring_params, self.top_scoring_row_data, self.all_row_data]

    def process(self):
        col_names = []
        indexes = []
        averaged_csv_data = []
        self.top_scoring_params.value = []
        self.all_row_data.value = []
        for i in range(len(self.all_p)):
            if self.hpam_model_type == "d2v":
                doc2vec_save = SaveLoad(rewrite=self.rewrite_model)
                identifier = "_WS_" + str(self.all_p[i]["window_size"]) + "_MC_" + \
                             str(self.all_p[i]["min_count"]) + "_TE_" + str(self.all_p[i]["train_epoch"]) + "_D_"+str(self.all_p[i]["dim"]) + "_D2V"
                doc2vec_fn = self.file_name + identifier
                d2v_classify_fn = self.classify_fn + identifier
                doc2vec_instance = d2v.D2V(self.all_p[i]["corpus_fn"], self.all_p[i]["wv_path"], doc2vec_fn,
                                           self.output_folder + "d2v/", doc2vec_save, self.all_p[i]["dim"], window_size=self.all_p[i]["window_size"],
                                           min_count=self.all_p[i]["min_count"], train_epoch=self.all_p[i]["train_epoch"]
                                           )
                doc2vec_instance.process_and_save()
                doc2vec_space = doc2vec_instance.getRep()

                split_ids = split.get_split_ids(self.data_type, self.matched_ids)
                x_train, y_train, x_test, y_test, x_dev, y_dev = split.split_data(doc2vec_space,
                                                                                  self.classes, split_ids,
                                                                                  dev_percent_of_train=self.dev_percent)
                hpam_save = SaveLoad(rewrite=self.rewrite_model)
                hyper_param = HParam(self.class_names,
                                     self.kfold_hpam_dict, self.model_type, d2v_classify_fn,
                                     self.output_folder, hpam_save, self.probability, rewrite_model=self.rewrite_model, x_train=x_train,
                                     y_train=y_train, x_test=x_test, y_test=y_test, x_dev=x_dev, y_dev=y_dev, final_score_on_dev=True, auroc=self.auroc)
                hyper_param.process_and_save()
                self.top_scoring_params.value.append(hyper_param.getTopScoringParams())
                self.all_row_data.append(hyper_param.getTopScoringRowData())
                top_scoring_row_data = hyper_param.getTopScoringRowData()
                averaged_csv_data.append(top_scoring_row_data[1])
                col_names = top_scoring_row_data[0]
                indexes.append(top_scoring_row_data[2][0])
        self.final_arrays.value = []
        self.final_arrays.value.append(col_names)
        self.final_arrays.value.append(np.asarray(averaged_csv_data).transpose())
        self.final_arrays.value.append(indexes)
        self.getTopScoringByMetric()
        super().process()

    #If you need other metrics than F1 just do metric = "acc" then index is 0 etc.
    def getTopScoringByMetric(self):
        doc2vec_space, index_sorted = self.getTopScoringSpace()
        split_ids = split.get_split_ids(self.data_type, self.matched_ids)
        x_train, y_train, x_test, y_test, x_dev, y_dev = split.split_data(doc2vec_space,
                                                                          self.classes, split_ids,
                                                                          dev_percent_of_train=self.dev_percent)
        model, model_fn = self.selectClassifier(self.top_scoring_params.value[index_sorted], x_train, y_train, x_test, y_test)

        score_save = SaveLoad(rewrite=self.rewrite_model, load_all = True)
        score = classify.selectScore(None, None, None, file_name=model_fn,
                                         output_folder=self.output_folder + "score/", save_class=score_save,
                                         verbose=True,
                                         fscore=self.fscore, acc=self.acc, kappa=self.kappa, auroc=self.auroc)
        score.process_and_save()

        score_dict = score.get()
        # This order cannot be changed as this is the order it is imported as.
        col_names = ["avg_acc", "avg_f1", "avg_kappa", "avg_prec", "avg_recall"]
        avg_array = [score_dict[col_names[0]], score_dict[col_names[1]],
                     score_dict[col_names[2]], score_dict[col_names[3]],
                     score_dict[col_names[4]]]
        self.top_scoring_row_data.value = [np.asarray(col_names), np.asarray(avg_array), np.asarray([model_fn])]

    def getTopScoringSpace(self):
        if self.final_arrays.value is None:
            self.final_arrays.value = self.save_class.load(self.final_arrays)
            self.top_scoring_params.value = self.save_class.load(self.top_scoring_params)
        list_of_scores = self.find_on_row_data(self.final_arrays.value)
        index_sorted = np.flipud(np.argsort(list_of_scores))[0]
        print("Training best parameters on test data not dev data")
        doc2vec_save = SaveLoad(rewrite=False)
        doc2vec_fn = self.file_name + "_WS_" + str(self.all_p[index_sorted]["window_size"]) + "_MC_" + str(
            self.all_p[index_sorted]["min_count"]) + "_TE_" + str(self.all_p[index_sorted]["train_epoch"]) + "_D_" + str(
            self.all_p[index_sorted]["dim"]) + "_D2V"

        doc2vec_instance = d2v.D2V(self.all_p[index_sorted]["corpus_fn"], self.all_p[index_sorted]["wv_path"], doc2vec_fn,
                                   self.output_folder + "d2v/", doc2vec_save, self.all_p[index_sorted]["dim"],
                                   window_size=self.all_p[index_sorted]["window_size"],
                                   min_count=self.all_p[index_sorted]["min_count"], train_epoch=self.all_p[index_sorted]["train_epoch"]
                                   )

        doc2vec_instance.process_and_save()
        doc2vec_space = doc2vec_instance.getRep()
        return doc2vec_space, index_sorted

class HParam(MasterHParam):

    # The CSV data that corresponds to the highest scoring row for the score_metric

    def __init__(self, class_names=None, hpam_dict=None, model_type=None, file_name=None, output_folder=None, save_class=None, probability=None, score_metric="avg_f1", rewrite_model=False, auroc=True, fscore=True, acc=True, kappa=True, x_train=None, y_train=None, x_test=None, y_test=None, x_dev=None, y_dev=None, final_score_on_dev=False,
                 mcm=None, dim_names=None):

        # Metric that determines what will be returned to the overall hyper-parameter method
        self.all_p = get_grid_params(hpam_dict)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.x_dev = x_dev
        self.final_score_on_dev = final_score_on_dev
        self.y_dev = y_dev
        self.file_names = []
        self.dim_names = dim_names

        super().__init__(rewrite_model=rewrite_model, auroc=auroc, fscore=fscore, acc=acc, kappa=kappa,
                         model_type=model_type, output_folder=output_folder,
                         file_name=file_name, probability=probability, class_names=class_names, save_class=save_class,
                         hpam_dict=hpam_dict,
                         score_metric=score_metric, mcm=mcm, dim_names=dim_names)
        self.end_file_name = self.file_name + "_Kfold" + str(self.dev) + str(
            generateNumber(hpam_dict)) + self.model_type


    def makePopos(self):
        self.averaged_csv_data = SaveLoadPOPO(self.averaged_csv_data, self.output_folder + "score/csv_averages/" + self.end_file_name + ".csv", "scoredictarray")
        self.top_scoring_row_data = SaveLoadPOPO(self.top_scoring_row_data, self.output_folder + "score/csv_averages/" + self.end_file_name + "Top"+self.score_metric+".csv", "csv")
        self.top_scoring_params = SaveLoadPOPO(self.top_scoring_params, self.output_folder + "score/csv_averages/top_params/" + self.end_file_name + "Top"+self.score_metric+".txt", "dct")

    def getTopScoringRowData(self):
        if self.processed is False:
            self.top_scoring_row_data.value = self.save_class.load(self.top_scoring_row_data)
        return self.top_scoring_row_data.value

    def getTopScoringParams(self):
        if self.processed is False:
            self.top_scoring_params.value = self.save_class.load(self.top_scoring_params)
        return self.top_scoring_params.value


    def makePopoArray(self):
        self.popo_array = [self.averaged_csv_data, self.top_scoring_row_data, self.top_scoring_params]

    def process(self):
        check_util.check_splits(self.x_train, self.y_train, self.x_test, self.y_test)
        check_util.check_splits(self.x_train, self.y_train, self.x_dev, self.y_dev)
        self.p_score_dicts = []
        for i in range(len(self.all_p)):
            model, model_fn = self.selectClassifier(self.all_p[i], self.x_train, self.y_train, self.x_dev, self.y_dev)
            pred, prob = self.trainClassifier(model)
            self.file_names.append(model_fn)
            score_save = SaveLoad(rewrite=self.rewrite_model, load_all=True)
            score = classify.selectScore(self.y_dev, pred, prob, file_name=model_fn,
                                             output_folder=self.output_folder + "score/", save_class=score_save, verbose=True,
                                             fscore=self.fscore, acc=self.acc, kappa=self.kappa, auroc=self.auroc, class_names = self.class_names)
            score.process_and_save()

            self.p_score_dicts.append(score.get())

        self.averaged_csv_data.value = [self.p_score_dicts, self.class_names, self.file_names, self.output_folder + "score/csv_details/"]
        self.getTopScoringByMetric()
        super().process()

    def getTopScoringByMetric(self):
        scores = []
        for i in range(len(self.p_score_dicts)):
            scores.append(self.p_score_dicts[i][self.score_metric])
        index_sorted = np.flipud(np.argsort(scores))[0]
        self.top_scoring_params.value = self.all_p[index_sorted]
        print("Training best parameters on test data not dev data")
        if self.final_score_on_dev:
            model, model_fn = self.selectClassifier(self.top_scoring_params.value, self.x_train, self.y_train, self.x_dev, self.y_dev)
        else:
            model, model_fn = self.selectClassifier(self.top_scoring_params.value, self.x_train, self.y_train,
                                                    self.x_test, self.y_test)
        pred, prob = self.trainClassifier(model)
        score_save = SaveLoad(rewrite=self.rewrite_model, load_all = True)
        if self.final_score_on_dev:
            score = classify.selectScore(self.y_dev, pred, prob, file_name=model_fn,
                                             output_folder=self.output_folder + "score/", save_class=score_save,
                                             verbose=True, class_names=self.class_names,
                                             fscore=self.fscore, acc=self.acc, kappa=self.kappa, auroc=self.auroc)
        else:
            score = classify.selectScore(self.y_test, pred, prob, file_name=model_fn,
                                             output_folder=self.output_folder + "score/", save_class=score_save,
                                             verbose=True, class_names = self.class_names,
                                             fscore=self.fscore, acc=self.acc, kappa=self.kappa, auroc=self.auroc)

        score.process_and_save()

        score_dict = score.get()
        col_names = ["avg_acc", "avg_f1", "avg_kappa", "avg_prec", "avg_recall"]
        avg_array = [score_dict[col_names[0]], score_dict[col_names[1]],
                     score_dict[col_names[2]], score_dict[col_names[3]],
                     score_dict[col_names[4]]]
        self.top_scoring_row_data.value = [col_names, avg_array, [model_fn]]


from util import py

def generateNumber(hpam_dict):
    try:
        hpam_dict["wv_path"][0] = "D" + hpam_dict["wv_path"][0][1:]
    except KeyError:
        print("wv path didnt exist")
    hyperparams_array = list(hpam_dict.values())
    hyperparam_names = list(hpam_dict.keys())
    unique_number = 0
    all_names = ""
    for i in range(len(hyperparams_array)):
        names_val = np.sum([ord(c) for c in hyperparam_names[i]])
        unique_number += names_val
        all_names += hyperparam_names[i]
        for j in range(len(hyperparams_array[i])):
            if py.isFloat(hyperparams_array[i][j]):
                unique_number += float(hyperparams_array[i][j])
            elif hyperparams_array[i][j] == None:
                unique_number += 3
            elif type(hyperparams_array[i][j]) is str:
                val = np.sum([ord(c) for c in hyperparams_array[i][j]])
                unique_number += val
    unique_number = unique_number * (len(hyperparam_names) / 10)
    while (float(unique_number)).is_integer() is False:
        unique_number *= 10
    print("Unique number is", unique_number)
    return int(unique_number)


