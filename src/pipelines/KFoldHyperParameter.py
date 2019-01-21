
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
from util import check
from rep import d2v
from model.randomforest import RandomForest
from util import split
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

    def __init__(self, class_names=None, hpam_dict=None, model_type=None, file_name=None, output_folder=None, save_class=None, probability=None, score_metric="avg_f1", rewrite_model=False, auroc=True, fscore=True, acc=True, kappa=True):

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
        self.file_names = []
        self.end_file_name = self.file_name + "_Kfold" + str(self.dev) + str(generateNumber(hpam_dict)) + self.model_type
        super().__init__(file_name, save_class)

    def trainClassifier(self, model):
        model.process_and_save()
        pred = model.test_predictions.value
        prob = None
        if self.probability:
            prob = model.test_proba.value
        return pred, prob

    def selectClassifier(self, all_p, x_train, y_train, x_test, y_test):
        svm_save = SaveLoad(rewrite=self.rewrite_model)
        model = None
        model_fn = None
        if self.model_type == "LinearSVM":
            model_fn = self.file_name + "_Dev" + "_" + str(len(x_test)) + "_Balanced_" + str(all_p["class_weight"]) \
                       + "_C_" + str(all_p["C"]) + "_Prob_" + str(self.probability) + "_" + self.model_type
            model = LinearSVM(x_train, y_train, x_test, y_test,
                              self.output_folder + "rep/svm/" + model_fn, svm_save, C=all_p["C"],
                              class_weight=all_p["class_weight"], probability=self.probability, verbose=False)

        elif self.model_type == "GaussianSVM":
            param_fn = "_Balanced_" + str(all_p["class_weight"]) \
                       + "_C_" + str(all_p["C"]) + "_Gam_" + str(all_p["gamma"]) + "_Prob_" + str(
                self.probability) + "_" + self.model_type

            model_fn = self.file_name + "_Dev"+ "_" + str(len(x_test))  + param_fn

            model = GaussianSVM(x_train, y_train, x_test, y_test,
                                self.output_folder + "rep/svm/" + model_fn, svm_save, gamma=all_p["gamma"],
                                C=all_p["C"],
                                class_weight=all_p["class_weight"], probability=self.probability, verbose=False)
        elif self.model_type == "RandomForest":
            param_fn = "MClass_Balanced_" + str(all_p["class_weight"]) \
                       + "_Estim_" + str(all_p["n_estimators"]) + "_Features_" + str(
                all_p["max_features"]) + "_BS_" + str(all_p["bootstrap"]) + "_MD_" + str(
                all_p["max_depth"]) + "_MSL_" + str(all_p["min_samples_leaf"]) + "_MSS_" + str(
                all_p["min_samples_split"]) + "_" + self.model_type

            model_fn = self.file_name + "_Dev"+ "_" + str(len(x_test))  + param_fn
            model = RandomForest(x_train, y_train, x_test, y_test,
                                 self.output_folder + "rep/svm/" + model_fn, svm_save,
                                 n_estimators=all_p["n_estimators"], bootstrap=all_p["bootstrap"],
                                 max_depth=all_p["max_depth"],
                                 min_samples_leaf=all_p["min_samples_leaf"],
                                 min_samples_split=all_p["min_samples_split"],
                                 class_weight=all_p["class_weight"], max_features=all_p["max_features"],
                                 probability=self.probability, verbose=False)
        return model, model_fn

# There's a way bette rto do all of this but I just pushed it through for time's sake.
class RecHParam(MasterHParam):

    classes = None
    folds = None
    file_name = None
    space = None
    output_folder = None
    model_type = None
    hyperparam_names = None
    hyperparams_array = None
    end_file_name = ""
    class_names = None
    probability = None
    averaged_csv_data = None
    average_file_names = []
    all_p = None
    auroc = False
    fscore = False
    acc = False
    kappa = False
    rewrite_model = False
    hpam_dict = None
    hpam_model_type = None
    final_arrays = None
    kfold_hpam_dict = None
    dev_percent = None
    data_type = None
    top_scoring_params = None

    def __init__(self, space, classes, class_names, hpam_dict, kfold_hpam_dict, hpam_model_type, model_type, file_name, output_folder, save_class, probability, rewrite_model=False, auroc=True, fscore=True, acc=True, kappa=True, dev_percent=0.2, score_metric=None, data_type=None):
        self.kfold_hpam_dict = kfold_hpam_dict
        self.hpam_model_type = hpam_model_type
        self.dev_percent=dev_percent
        self.data_type = data_type
        self.all_p = get_grid_params(hpam_dict)
        self.space = space
        self.classes = classes
        self.average_file_names = []
        self.end_file_name = self.file_name + "_Kfold" + str(self.folds) + str(generateNumber(hpam_dict)) + self.model_type
        super().__init__(rewrite_model=rewrite_model, auroc=auroc, fscore=fscore, acc=acc, kappa=kappa, model_type=model_type, output_folder=output_folder,
                         file_name=file_name, probability=probability, class_names=class_names,  save_class=save_class, hpam_dict=hpam_dict,
                         score_metric=score_metric)

    def makePopos(self):
        self.final_arrays = SaveLoadPOPO(self.final_arrays, self.output_folder + "rep/score/csv_averages/" + self.end_file_name + ".csv", "csv")


    def makePopoArray(self):
        self.popo_array = [self.final_arrays]

    def process(self):
        col_names = []
        indexes = []
        averaged_csv_data = []
        for i in range(len(self.all_p)):
            if self.hpam_model_type == "d2v":
                doc2vec_save = SaveLoad(rewrite=False)

                doc2vec_fn = self.file_name + "_WS_" + str(self.all_p[i]["window_size"]) + "_MC_" + str(self.all_p[i]["min_count"]) + "_TE_" + str(self.all_p[i]["train_epoch"]) + "_D_"+str(self.all_p[i]["dim"]) + "_D2V"
                doc2vec_instance = d2v.D2V(self.all_p[i]["corpus_fn"], self.all_p[i]["wv_path"], doc2vec_fn,
                                           self.output_folder + "rep/d2v/", doc2vec_save, self.all_p[i]["dim"], window_size=self.all_p[i]["window_size"],
                                           min_count=self.all_p[i]["min_count"], train_epoch=self.all_p[i]["train_epoch"]
                                           )
                doc2vec_instance.process_and_save()
                doc2vec_space = doc2vec_instance.rep.value

                split_ids = split.get_split_ids(self.data_type)
                x_train, y_train, x_test, y_test, x_dev, y_dev = split.split_data(doc2vec_space,
                                                                                  self.classes, split_ids,
                                                                                  dev_percent_of_train=self.dev_percent)
                hpam_save = SaveLoad(rewrite=self.rewrite_model)
                hyper_param = HParam(self.class_names,
                                     self.kfold_hpam_dict, self.model_type, doc2vec_fn,
                                     self.output_folder, hpam_save, self.probability, rewrite_model=self.rewrite_model, x_train=x_train,
                                     y_train=y_train, x_test=x_test, y_test=y_test, x_dev=x_dev, y_dev=y_dev, final_score_on_dev=True)

                self.top_scoring_params = hyper_param.top_scoring_params
                hyper_param.process_and_save()
                if hyper_param.top_scoring_row_data.value[1][1] < 0.7:
                    raise ValueError("Accuracy is below 0.7, potential mismatch")
                averaged_csv_data.append(hyper_param.top_scoring_row_data.value[1])
                col_names = hyper_param.top_scoring_row_data.value[0]
                indexes.append(hyper_param.top_scoring_row_data.value[2][0])
        self.final_arrays.value = []
        self.final_arrays.value.append(col_names)
        self.final_arrays.value.append(np.asarray(averaged_csv_data).transpose())
        self.final_arrays.value.append(indexes)
        self.getTopScoringByMetric()

    #If you need other metrics than F1 just do metric = "acc" then index is 0 etc.
    def getTopScoringByMetric(self):
        list_of_f1_scores = self.final_arrays.value[1][0]
        index_sorted = np.flipud(np.argsort(list_of_f1_scores))
        print("Training best parameters on test data not dev data")
        doc2vec_save = SaveLoad(rewrite=False)

        doc2vec_fn = self.file_name + "_WS_" + str(self.all_p[index_sorted]["window_size"]) + "_MC_" + str(
            self.all_p[index_sorted]["min_count"]) + "_TE_" + str(self.all_p[index_sorted]["train_epoch"]) + "_D_" + str(
            self.all_p[index_sorted]["dim"]) + "_D2V"

        doc2vec_instance = d2v.D2V(self.all_p[index_sorted]["corpus_fn"], self.all_p[index_sorted]["wv_path"], doc2vec_fn,
                                   self.output_folder + "rep/d2v/", doc2vec_save, self.all_p[index_sorted]["dim"],
                                   window_size=self.all_p[index_sorted]["window_size"],
                                   min_count=self.all_p[index_sorted]["min_count"], train_epoch=self.all_p[index_sorted]["train_epoch"]
                                   )

        doc2vec_instance.process_and_save()
        doc2vec_space = doc2vec_instance.rep.value

        split_ids = split.get_split_ids(self.data_type)
        x_train, y_train, x_test, y_test, x_dev, y_dev = split.split_data(doc2vec_space,
                                                                          self.classes, split_ids,
                                                                          dev_percent_of_train=self.dev_percent)
        model, model_fn = self.selectClassifier(self.top_scoring_params, x_train, y_train, x_test, y_test)
        pred, prob = self.trainClassifier(model)
        score_save = SaveLoad(rewrite=self.rewrite_model)
        score = classify.MultiClassScore(y_test, pred, prob, file_name=model_fn,
                                         output_folder=self.output_folder + "rep/score/", save_class=score_save,
                                         verbose=True,
                                         fscore=self.fscore, acc=self.acc, kappa=self.kappa, auroc=self.auroc)
        score.process_and_save()

        score_dict = score.get()
        col_names = ["avg_f1", "avg_acc", "avg_kappa", "avg_prec", "avg_recall"]
        avg_array = [score_dict[col_names[0]], score_dict[col_names[1]],
                     score_dict[col_names[2]], score_dict[col_names[3]],
                     score_dict[col_names[4]]]
        self.top_scoring_row_data.value = [col_names, avg_array, [model_fn]]



class HParam(MasterHParam):

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

    def __init__(self, class_names, hpam_dict, model_type, file_name, output_folder, save_class, probability, score_metric="avg_f1", rewrite_model=False, auroc=True, fscore=True, acc=True, kappa=True, x_train=None, y_train=None, x_test=None, y_test=None, x_dev=None, y_dev=None, final_score_on_dev=False):

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
        self.end_file_name = self.file_name + "_Kfold" + str(self.dev) + str(generateNumber(hpam_dict)) + self.model_type
        check.check_splits(self.x_train, self.y_train, self.x_test, self.y_test)
        check.check_splits(self.x_train, self.y_train, self.x_dev, self.y_dev)
        super().__init__(rewrite_model=rewrite_model, auroc=auroc, fscore=fscore, acc=acc, kappa=kappa,
                         model_type=model_type, output_folder=output_folder,
                         file_name=file_name, probability=probability, class_names=class_names, save_class=save_class,
                         hpam_dict=hpam_dict,
                         score_metric=score_metric)


    def makePopos(self):
        self.averaged_csv_data = SaveLoadPOPO(self.averaged_csv_data, self.output_folder + "rep/score/csv_averages/" + self.end_file_name + ".csv", "scoredictarray")
        self.top_scoring_row_data = SaveLoadPOPO(self.top_scoring_row_data, self.output_folder + "rep/score/csv_averages/" + self.end_file_name + "Top"+self.score_metric+".csv", "csv")

    def makePopoArray(self):
        self.popo_array = [self.averaged_csv_data, self.top_scoring_row_data]

    def process(self):
        self.p_score_dicts = []
        for i in range(len(self.all_p)):
            model, model_fn = self.selectClassifier(self.all_p[i], self.x_train, self.y_train, self.x_dev, self.y_dev)
            pred, prob = self.trainClassifier(model)
            self.file_names.append(model_fn)
            score_save = SaveLoad(rewrite=self.rewrite_model)
            score = classify.MultiClassScore(self.y_dev, pred, prob, file_name=model_fn,
                                             output_folder=self.output_folder + "rep/score/", save_class=score_save, verbose=True,
                                             fscore=self.fscore, acc=self.acc, kappa=self.kappa, auroc=self.auroc)
            score.process_and_save()

            self.p_score_dicts.append(score.get())

        self.averaged_csv_data.value = [self.p_score_dicts, self.class_names, self.file_names, self.output_folder + "rep/score/csv_details/"]
        self.getTopScoringByMetric()

    def getTopScoringByMetric(self):
        scores = []
        for i in range(len(self.p_score_dicts)):
            scores.append(self.p_score_dicts[i][self.score_metric])
        index_sorted = np.flipud(np.argsort(scores))[0]
        self.top_scoring_params = self.all_p[index_sorted]
        print("Training best parameters on test data not dev data")
        model, model_fn = self.selectClassifier(self.top_scoring_params, self.x_train, self.y_train, self.x_test, self.y_test)
        pred, prob = self.trainClassifier(model)
        score_save = SaveLoad(rewrite=self.rewrite_model)
        if self.final_score_on_dev:
            score = classify.MultiClassScore(self.y_test, pred, prob, file_name=model_fn,
                                             output_folder=self.output_folder + "rep/score/", save_class=score_save,
                                             verbose=True,
                                             fscore=self.fscore, acc=self.acc, kappa=self.kappa, auroc=self.auroc)
        else:
            score = classify.MultiClassScore(self.y_dev, pred, prob, file_name=model_fn,
                                             output_folder=self.output_folder + "rep/score/", save_class=score_save,
                                             verbose=True,
                                             fscore=self.fscore, acc=self.acc, kappa=self.kappa, auroc=self.auroc)

        score.process_and_save()

        score_dict = score.get()
        col_names = ["avg_f1", "avg_acc", "avg_kappa", "avg_prec", "avg_recall"]
        avg_array = [score_dict[col_names[0]], score_dict[col_names[1]],
                     score_dict[col_names[2]], score_dict[col_names[3]],
                     score_dict[col_names[4]]]
        self.top_scoring_row_data.value = [col_names, avg_array, [model_fn]]

#### NEEDS REWRITE TO RE-USE OVERALL HYPERPARAMTER THING
class KFoldHyperParameter(Method):

    classes = None
    folds = None
    file_name = None
    space = None
    output_folder = None
    model_type = None
    end_file_name = ""
    class_names = None
    probability = None
    averaged_csv_data = None
    average_file_names = []
    all_p = None
    auroc = False
    fscore = False
    acc = False
    kappa = False
    rewrite_model = False
    hpam_dict = None
    score_metric = None
    # The CSV data that corresponds to the highest scoring row for the score_metric
    top_scoring_row_data = None

    def __init__(self, space, classes, class_names, hpam_dict, model_type, file_name, output_folder, save_class, probability, score_metric="avg_f1", rewrite_model=False, auroc=True, fscore=True, acc=True, kappa=True, folds=2):

        self.rewrite_model = rewrite_model
        # Metric that determines what will be returned to the overall hyper-parameter method
        self.score_metric = score_metric
        self.classes = classes
        self.class_names = class_names
        self.probability = probability
        self.hpam_dict = hpam_dict
        self.auroc = auroc
        self.fscore = fscore
        self.acc = acc
        self.kappa = kappa
        self.all_p = get_grid_params(hpam_dict)
        self.folds = folds
        self.file_name = file_name
        self.space = space
        self.output_folder = output_folder
        self.model_type = model_type
        self.average_file_names = []
        self.end_file_name = self.file_name + "_Kfold" + str(self.folds) + str(generateNumber(hpam_dict)) + self.model_type
        super().__init__(file_name, save_class)

    def makePopos(self):
        self.averaged_csv_data = SaveLoadPOPO(self.averaged_csv_data, self.output_folder + "rep/score/csv_averages/" + self.end_file_name + ".csv", "scoredictarray")
        self.top_scoring_row_data = SaveLoadPOPO(self.top_scoring_row_data, self.output_folder + "rep/score/csv_averages/" + self.end_file_name + "Top"+self.score_metric+".csv", "csv")

    def makePopoArray(self):
        self.popo_array = [self.averaged_csv_data, self.top_scoring_row_data]

    def process(self):
        check.check_x(self.space)
        kf = KFold(n_splits=self.folds)
        fold_num = 0
        to_average_score_dicts = []

        for train_index, test_index in kf.split(self.space):

            print("TRAIN:", train_index, "TEST:", test_index)

            x_train, x_test = self.space[train_index], self.space[test_index]
            y_train, y_test = self.classes[train_index], self.classes[test_index]

            p_score_dicts = []
            for i in range(len(self.all_p)):
                svm_save = SaveLoad(rewrite=self.rewrite_model)
                if self.model_type == "LinearSVM":
                    model_fn = self.file_name + "_Kfold" + str(fold_num) + "_Balanced_" + str(self.all_p[i]["class_weight"])\
                               + "_C_" + str(self.all_p[i]["C"]) + "_Prob_" + str(self.probability) + "_" + self.model_type
                    average_fn = self.file_name + "_Kfold" + str(self.folds) + "_Balanced_" + str(self.all_p[i]["class_weight"])\
                               + "_C_" + str(self.all_p[i]["C"])  + "_Prob_" + str(self.probability) +  self.model_type
                    model = LinearSVM(x_train, y_train, x_test, y_test,
                                    self.output_folder + "rep/svm/" + model_fn, svm_save, C=self.all_p[i]["C"],
                                    class_weight=self.all_p[i]["class_weight"], probability=self.probability, verbose=False)

                elif self.model_type == "GaussianSVM":
                    param_fn = "_Balanced_" + str(self.all_p[i]["class_weight"])\
                               + "_C_" +str(self.all_p[i]["C"]) + "_Gam_"+ str(self.all_p[i]["gamma"]) + "_Prob_" + str(self.probability) + "_" + self.model_type

                    model_fn = self.file_name + "Kfold" + str(fold_num) + param_fn
                    average_fn = self.file_name + "Kfold" + str(self.folds) + param_fn

                    model = GaussianSVM(x_train, y_train, x_test, y_test,
                                    self.output_folder + "rep/svm/" + model_fn, svm_save, gamma=self.all_p[i]["gamma"], C=self.all_p[i]["C"],
                                    class_weight=self.all_p[i]["class_weight"], probability=self.probability, verbose=False)
                elif self.model_type == "RandomForest":
                    param_fn = "MClass_Balanced_" + str(self.all_p[i]["class_weight"])\
                               + "_Estim_" +str(self.all_p[i]["n_estimators"]) + "_Features_"+ str(self.all_p[i]["max_features"])   + "_BS_"+ str(self.all_p[i]["bootstrap"]) + "_MD_" +str(self.all_p[i]["max_depth"])+"_MSL_"+ str(self.all_p[i]["min_samples_leaf"]) + "_MSS_"+str(self.all_p[i]["min_samples_split"])+ "_" + self.model_type

                    model_fn = self.file_name + "Kfold" + str(fold_num) + param_fn
                    average_fn = self.file_name + "Kfold" + str(self.folds) + param_fn
                    model = RandomForest(x_train, y_train, x_test, y_test,
                                    self.output_folder + "rep/svm/" + model_fn, svm_save, n_estimators=self.all_p[i]["n_estimators"], bootstrap = self.all_p[i]["bootstrap"], max_depth = self.all_p[i]["max_depth"],
                                         min_samples_leaf=self.all_p[i]["min_samples_leaf"], min_samples_split=self.all_p[i]["min_samples_split"],
                                    class_weight=self.all_p[i]["class_weight"], max_features=self.all_p[i]["max_features"], probability=self.probability, verbose=False)
                model.process_and_save()
                pred = model.test_predictions.value
                prob = None
                if self.probability:
                    prob = model.test_proba.value
                if fold_num == 0:
                    self.average_file_names.append(average_fn)
                score_save = SaveLoad(rewrite=self.rewrite_model)
                score = classify.MultiClassScore(y_test, pred, prob, file_name=model_fn,
                                                 output_folder=self.output_folder + "rep/score/", save_class=score_save, verbose=True,
                                                 fscore=self.fscore, acc=self.acc, kappa=self.kappa, auroc=self.auroc)
                score.process_and_save()

                p_score_dicts.append(score.get())

            fold_num += 1
            to_average_score_dicts.append(p_score_dicts)
        self.to_average_score_dicts = np.asarray(to_average_score_dicts).transpose()
        self.averaged_param_score_dicts = []
        for i in range(int(len(self.to_average_score_dicts))):
            averaged_score_dict = classify.average_scores(self.to_average_score_dicts[i])
            self.averaged_param_score_dicts.append(averaged_score_dict)
        self.averaged_csv_data.value = [self.averaged_param_score_dicts, self.class_names, self.average_file_names, self.output_folder + "rep/score/csv_details/"]
        self.getTopScoringByMetric()


    def getTopScoringByMetric(self):
        scores = []
        for i in range(len(self.averaged_param_score_dicts)):
            scores.append(self.averaged_param_score_dicts[i][self.score_metric])
        index_sorted = np.flipud(np.argsort(scores))[0]
        col_names = ["avg_f1", "avg_acc", "avg_kappa", "avg_prec", "avg_recall"]
        avg_array = [self.averaged_param_score_dicts[index_sorted][col_names[0]], self.averaged_param_score_dicts[index_sorted][col_names[1]],
                     self.averaged_param_score_dicts[index_sorted][col_names[2]], self.averaged_param_score_dicts[index_sorted][col_names[3]],
                     self.averaged_param_score_dicts[index_sorted][col_names[4]]]
        self.top_scoring_row_data.value = [col_names, avg_array, [self.average_file_names[index_sorted]]]


def generateNumber(hpam_dict):
    hyperparams_array = list(hpam_dict.values())
    hyperparam_names = list(hpam_dict.keys())
    unique_number = 0
    all_names = ""
    for i in range(len(hyperparams_array)):
        names_val = np.sum([ord(c) for c in hyperparam_names[i]])
        unique_number += names_val
        all_names += hyperparam_names[i]
    unique_number = unique_number * (len(hyperparam_names) / 10)
    while (float(unique_number)).is_integer() is False:
        unique_number *= 10
    print("Unique number is", unique_number)
    return int(unique_number)


