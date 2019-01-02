
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

class HyperParameter(Method):

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

    def __init__(self, space, classes, class_names, hyperparams_array, hyperparam_names, model_type, file_name, output_folder, save_class, probability, auroc=True, fscore=True, acc=True, kappa=True, folds=2):
        self.classes = classes
        self.class_names = class_names
        self.probability = probability
        self.hyperparams_array = hyperparams_array
        self.auroc = auroc
        self.fscore = fscore
        self.acc = acc
        self.kappa = kappa
        all_params = list(
            product(
                *hyperparams_array
            )
        )
        self.hyperparam_names = hyperparam_names
        self.all_p = []
        for i in range(len(all_params)):
            pdict = {}
            for j in range(len(self.hyperparam_names)):
                pdict[self.hyperparam_names[j]] = all_params[i][j]
            self.all_p.append(pdict)

        self.folds = folds
        self.file_name = file_name
        self.space = space
        self.output_folder = output_folder
        self.model_type = model_type
        self.average_file_names = []
        self.end_file_name = self.file_name + "_Kfold" + str(self.folds) + str(generateNumber(self.hyperparams_array, self.hyperparam_names)) + self.model_type
        super().__init__(file_name, save_class)

    def makePopos(self):
        self.averaged_csv_data = SaveLoadPOPO(self.averaged_csv_data, self.output_folder + "rep/score/csv_averages/" + self.end_file_name + ".csv", "scoredictarray")

    def makePopoArray(self):
        self.popo_array = [self.averaged_csv_data]

    def process(self):

        kf = KFold(n_splits=self.folds)
        fold_num = 0
        to_average_score_dicts = []

        for train_index, test_index in kf.split(self.space):

            print("TRAIN:", train_index, "TEST:", test_index)

            x_train, x_test = self.space[train_index], self.space[test_index]
            y_train, y_test = self.classes[train_index], self.classes[test_index]

            p_score_dicts = []
            for i in range(len(self.all_p)):
                svm_save = SaveLoad(rewrite=True)
                if self.model_type == "LinearSVM":
                    model_fn = self.file_name + "_Kfold" + str(fold_num) + "_Balanced_" + str(self.all_p[i]["class_weight"])\
                               + "_C_" + str(self.all_p[i]["C"]) + "_Prob_" + str(self.probability) + "_" + self.model_type
                    average_fn = self.file_name + "_Kfold" + str(self.folds) + "_Balanced_" + str(self.all_p[i]["class_weight"])\
                               + "_C_" + str(self.all_p[i]["C"])  + self.model_type
                    model = LinearSVM(x_train, y_train, x_test, y_test,
                                    self.output_folder + "rep/svm/" + model_fn, svm_save, C=self.all_p[i]["C"],
                                    class_weight=self.all_p[i]["class_weight"], probability=self.probability, verbose=False)

                elif self.model_type == "GaussianSVM":
                    model_fn = self.file_name + "_Kfold" + str(fold_num) + "_Balanced_" + str(self.all_p[i]["class_weight"])\
                               + "_C_" +self.all_p[i]["C"] + "_Gam_"+ self.all_p[i]["gamma"] + "_Prob_" + str(self.probability) + "_" + self.model_type
                    average_fn = self.file_name + "_Kfold" + str(self.folds) + "_Balanced_" + str(self.all_p[i]["class_weight"])\
                               + "_C_" + str(self.all_p[i]["C"])  + self.model_type
                    model = GaussianSVM(x_train, y_train, x_test, y_test,
                                    self.output_folder + "rep/svm/" + model_fn, svm_save, gamma=self.all_p[i]["gamma"], C=self.all_p[i]["C"],
                                    class_weight=self.all_p[i]["class_weight"], probability=self.probability, verbose=False)
                model.process_and_save()
                pred = model.test_predictions.value
                prob = None
                if self.probability:
                    prob = model.test_proba.value
                if fold_num == 0:
                    self.average_file_names.append(average_fn)
                score_save = SaveLoad(rewrite=True)
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

    def process_and_save(self):
        if os.path.exists(self.end_file_name) is False:
            print(self.__class__.__name__, "Doesn't exist, creating")
            self.process()
            self.save()
            print("corpus done")
        else:
            print(self.__class__.__name__, "Already exists")


def generateNumber(hyperparams_array, hyperparam_names):
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


