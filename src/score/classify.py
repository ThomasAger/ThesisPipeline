
import math
import os.path

from score.functions import get_f1_score
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score
from sklearn.metrics import precision_recall_fscore_support
from common import Method
from util import io as dt
from common.SaveLoadPOPO import SaveLoadPOPO
from util import check_util
from util import py
import scipy.sparse as sp

def all_scores(true_target, prediction):

    prec, recall, fbeta, score = precision_recall_fscore_support(true_target, prediction,
                                                                     average="binary")
    auroc = roc_auc_score(true_target, prediction, average="macro")
    try:
        f1 = get_f1_score(prec, recall)
    except ZeroDivisionError:
        f1 = 0.0
    # For multi-label, must exactly match
    accuracy = accuracy_score(true_target, prediction)
    if math.isnan(f1):
        f1 = 0.0
    if math.isnan(accuracy):
        f1 = 0.0
    if math.isnan(prec):
        f1 = 0.0
    if math.isnan(recall):
        f1 = 0.0
    kappa = cohen_kappa_score(true_target, prediction)
    return f1, prec, recall, accuracy, auroc, kappa


class MasterScore(Method.Method):

    true_targets = None
    predictions = None
    pred_proba = None
    output_folder = None
    class_names = None
    csv_data = None
    score_dict = None

    f1s = None
    precs = None
    recalls = None
    accs = None
    aurocs = None
    kappas = None

    avg_f1 = None
    avg_acc = None
    avg_auroc = None
    avg_kappa = None
    avg_prec = None
    avg_recall = None
    verbose = None

    auroc = False
    fscore = False
    kappa = False
    acc = False
    f1 = False
    save_csv = False

    def __init__(self, true_targets, predictions, pred_proba,  file_name, output_folder, save_class, f1=True, auroc=True,
                 fscore=True, kappa=True, acc=True, class_names=None, verbose=True, save_csv=False):
        self.save_csv = save_csv
        super().__init__(file_name, save_class)

    def process(self):
        check_util.check_y(self.true_targets, self.predictions)
        if np.count_nonzero(self.true_targets) < 1:
            auroc = False
            print("Auroc has been automatically disabled as the true targets (len)", len(self.true_targets),
                  "are all zero. (It causes an error)")
        if self.pred_proba is None:
            self.auroc = False
            print("Auroc has been automatically disabled as probabilities were not provided")

        if sp.issparse(self.predictions):
            print("Shape is:", self.predictions.shape)
        else:
            if py.isArray(self.predictions[0]):
                print("Shape is:", len(self.predictions), len(self.predictions[0]))
            else:
                print("Shape is:", len(self.predictions))
        if self.auroc:
            self.calc_auroc()
        if self.f1:
            self.calc_fscore()
        if self.kappa:
            self.calc_kappa()
        if self.acc:
            self.calc_acc()
        self.checkScores()
        if self.save_csv:
            self.csv_data.value = [self.get(), self.class_names]
        if self.verbose:
            self.print()
        super().process()

    def checkScores(self):
        if self.f1 and math.isnan(self.avg_f1.value) or self.f1 and math.isnan(self.f1s.value[0]):
            raise ValueError("F1 is NaN, Array", self.f1s.value, "Average", self.avg_f1.value)
        if self.auroc and math.isnan(self.avg_auroc.value) or self.auroc and math.isnan(self.aurocs.value[0]):
            raise ValueError("Auroc is NaN, Array", self.aurocs.value, "Average", self.avg_auroc.value)
        if self.kappa and math.isnan(self.avg_kappa.value) or self.kappa and math.isnan(self.kappas.value[0]):
            raise ValueError("Kappa is NaN, Array", self.kappas.value, "Average", self.avg_kappa.value)
        if self.acc and math.isnan(self.avg_acc.value) or self.acc and math.isnan(self.accs.value[0]):
            raise ValueError("Acc is NaN, Array", self.accs.value, "Average", self.avg_acc.value)

    def makePopos(self):
        included_scores = ""
        if self.f1:
            self.f1s = SaveLoadPOPO(np.full(len(self.true_targets), math.nan), self.output_folder + "f1/" + self.file_name + "_F1.txt", "1dtxtf")
            self.avg_f1 = SaveLoadPOPO(self.avg_f1, self.output_folder + "f1/" + self.file_name + "_Avg_F1.txt", "txtf")
            self.precs = SaveLoadPOPO(np.full(len(self.true_targets), math.nan), self.output_folder + "prec/" + self.file_name + "_Prec.txt", "1dtxtf")
            self.avg_prec = SaveLoadPOPO(self.avg_prec, self.output_folder + "prec/" + self.file_name + "_avg_prec.txt", "txtf")
            self.recalls = SaveLoadPOPO(np.full(len(self.true_targets), math.nan), self.output_folder + "recall/" + self.file_name + "_Recall.txt", "1dtxtf")
            self.avg_recall = SaveLoadPOPO(self.avg_recall, self.output_folder + "recall/" + self.file_name + "_avg_recall.txt", "txtf")
            included_scores += "F1_"
        if self.acc:
            self.accs = SaveLoadPOPO(np.full(len(self.true_targets), math.nan), self.output_folder + "acc/" + self.file_name + "_Acc.txt", "1dtxtf")
            self.avg_acc = SaveLoadPOPO(self.avg_acc, self.output_folder + "acc/" + self.file_name + "_avg_acc.txt", "txtf")
            included_scores += "ACC_"
        if self.auroc:
            self.aurocs = SaveLoadPOPO(np.full(len(self.true_targets), math.nan), self.output_folder + "auroc/" + self.file_name + "_Auroc.txt", "1dtxtf")
            self.avg_auroc = SaveLoadPOPO(self.avg_auroc, self.output_folder + "auroc/" + self.file_name + "_avg_auroc.txt", "txtf")
            included_scores += "AUROC_"
        if self.kappa:
            self.kappas = SaveLoadPOPO(np.full(len(self.true_targets), math.nan), self.output_folder + "kappa/" + self.file_name + "_Kappa.txt", "1dtxtf")
            self.avg_kappa = SaveLoadPOPO(self.avg_kappa, self.output_folder + "kappa/" + self.file_name + "_avg_kappa.txt", "txtf")
            included_scores += "Kappa_"
        if self.save_csv:
            self.csv_data = SaveLoadPOPO(self.csv_data , self.output_folder + "csv_details/" + self.file_name + "_"+included_scores+".csv", "scoredict")

    def makePopoArray(self):
        self.popo_array = []
        if self.f1:
            self.popo_array.append(self.f1s)
            self.popo_array.append(self.precs)
            self.popo_array.append(self.recalls)
            self.popo_array.append(self.avg_f1)
            self.popo_array.append(self.avg_prec)
            self.popo_array.append(self.avg_recall)

        if self.acc:
            self.popo_array.append(self.accs)
            self.popo_array.append(self.avg_acc)

        if self.auroc:
            self.popo_array.append(self.aurocs)
            self.popo_array.append(self.avg_auroc)

        if self.kappa:
            self.popo_array.append(self.kappas)
            self.popo_array.append(self.avg_kappa)

        if self.save_csv:
            self.popo_array.append(self.csv_data)

    def get(self):
        score_dict = {}
        if self.f1:
            score_dict["f1"] = self.f1s.value
            score_dict["avg_f1"] = self.avg_f1.value
            score_dict["prec"] = self.precs.value
            score_dict["avg_prec"] = self.avg_prec.value
            score_dict["recall"] = self.recalls.value
            score_dict["avg_recall"] = self.avg_recall.value

        if self.acc:
            score_dict["acc"] = self.accs.value
            score_dict["avg_acc"] =  self.avg_acc.value

        if self.auroc:
            score_dict["auroc"] = self.aurocs.value
            score_dict["avg_auroc"] = self.avg_auroc.value

        if self.kappa:
            score_dict["kappa"] = self.kappas.value
            score_dict["avg_kappa"] = self.avg_kappa.value
        return score_dict

    def loadScores(self):
        self.popo_array = self.save_class.loadAll(self.popo_array)

    def print(self, max_to_print=5):
        score_dict = {}
        print("Amt", len(self.f1s.value))

        if self.f1:
            print("prec", self.precs.value[:max_to_print])
            print("recall",  self.recalls.value[:max_to_print])
            print("f1s",  self.f1s.value[:max_to_print], "| average", self.avg_f1.value)

        if self.acc:
            print("acc", self.accs.value[:max_to_print], "| average", self.avg_acc.value)

        if self.auroc:
            print("auroc", self.aurocs.value[:max_to_print], "| average", self.avg_auroc.value)

        if self.kappa:
            print("kappa",  self.kappas.value[:max_to_print], "| average", self.avg_kappa.value)
        return score_dict

    def save(self):
        self.save_class.save(self.popo_array)

def selectScore(true_targets, predictions, pred_proba, file_name, output_folder, save_class, f1=True, auroc=False,
                 fscore=True, kappa=True, acc=True, class_names=None, verbose=True, save_csv=False, directions=False):
    if true_targets is None or py.isArray(true_targets[0]) and len(predictions) > 1:
        return MultiClassScore(true_targets, predictions, pred_proba, file_name, output_folder, save_class, f1=f1, auroc=auroc,
                 fscore=fscore, kappa=kappa, acc=acc, class_names=class_names, verbose=verbose, directions=directions, save_csv=save_csv)
    else:
        # Predictions are usually put into a class array, this circumvents that for single class arrays
        if len(predictions) == 1:
            predictions = predictions[0]
        return SingleClassScore(true_targets, predictions, pred_proba, file_name, output_folder, save_class, f1=f1, auroc=auroc,
                 fscore=fscore, kappa=kappa, acc=acc, class_names=class_names, verbose=verbose, directions=directions, save_csv=save_csv)

class MultiClassScore(MasterScore):
    def __init__(self, true_targets, predictions, pred_proba, file_name, output_folder, save_class, f1=True, auroc=True,
                 fscore=True, kappa=True, acc=True, class_names=None, verbose=True, directions=False, save_csv=False):

        self.true_targets = true_targets
        self.predictions = predictions
        self.pred_proba = pred_proba
        self.output_folder = output_folder
        self.class_names = class_names


        self.auroc = auroc
        self.fscore = fscore
        self.kappa = kappa
        self.acc = acc
        self.f1 = f1

        self.verbose = verbose
        if directions is False:
            self.true_targets = py.transIfRowsLarger(self.true_targets)
            self.predictions = py.transIfRowsLarger(self.predictions)
            self.pred_proba = py.transIfRowsLarger(self.pred_proba)

        if self.predictions is None:
            self.predictions = []
        if self.true_targets is None:
            self.true_targets = []
        if self.pred_proba is None:
            self.pred_proba = []
        super().__init__(true_targets, predictions, pred_proba, file_name, output_folder, save_class, f1=f1,
                         auroc=auroc,
                         fscore=fscore, kappa=kappa, acc=acc, class_names=class_names, verbose=verbose, save_csv=save_csv)
    def getPred(self, predict):
        if sp.issparse(predict):
            pred = np.asarray(predict.todense(), dtype=np.int32)[0]
        else:
            pred = predict
        return pred

    def getPredLen(self):
        if sp.issparse(self.predictions[0]):
            length = self.predictions.shape[0]
        else:
            length = len(self.predictions)
        return length

    def calc_fscore(self):
        print(len(self.true_targets))
        print(self.getPredLen())
        for i in range(self.getPredLen()):
            try:
                predict = self.getPred(self.predictions[i])
                self.precs.value[i], self.recalls.value[i], self.f1s.value[i], unused__ = precision_recall_fscore_support(
                    self.true_targets[i], predict, average="binary")
            except IndexError:
                print(i)

        if math.isnan(self.precs.value[i]):
            print("!!! WARNING !!!! precs is NaN")
            self.precs.value[i] = 0.0
        if math.isnan(self.recalls.value[i]):
            print("!!! WARNING !!!! recalls is NaN")
            self.recalls.value[i] = 0.0
        if math.isnan(self.f1s.value[i]):
            print("!!! WARNING !!!! f1s is NaN")
            self.f1s.value[i] = 0.0
        self.avg_prec.value = np.average(self.precs.value)
        self.avg_recall.value = np.average(self.recalls.value)
        self.avg_f1.value = get_f1_score(self.avg_prec.value, self.avg_recall.value)

    # Check different averages for auroc
    def calc_auroc(self):
        for i in range(len(self.predictions)):
            if sp.issparse(self.predictions[i]):
                self.predictions[i] = np.asarray(self.predictions[i].todense())
            self.aurocs.value[i] = roc_auc_score(self.true_targets[i], self.pred_proba[i])
            if math.isnan(self.aurocs.value[i]):
                print("!!! WARNING !!!! aurocs is NaN")
                self.aurocs.value[i] = 0.0
        self.avg_auroc.value = np.average(self.aurocs.value)

    def calc_acc(self):
        for i in range(self.getPredLen()):
            predict = self.getPred(self.predictions[i])
            self.accs.value[i] = accuracy_score(self.true_targets[i], predict)
            if math.isnan(self.accs.value[i]):
                print("!!! WARNING !!!! accs is NaN")
                self.accs.value[i] = 0.0
        self.avg_acc.value = np.average(self.accs.value)

    def calc_kappa(self):
        for i in range(self.getPredLen()):
            predict = self.getPred(self.predictions[i])
            self.kappas.value[i] = cohen_kappa_score(self.true_targets[i], predict)
            if math.isnan(self.kappas.value[i]):
                print("!!! WARNING !!!! kappas is NaN")
                self.kappas.value[i] = 0.0
        self.avg_kappa.value = np.average(self.kappas.value)

    def save(self):
        self.save_class.save(self.popo_array)

class SingleClassScore(MasterScore):


    def __init__(self, true_targets, predictions, pred_proba, file_name, output_folder, save_class, f1=True, auroc=False,
                 fscore=True, kappa=True, acc=True, class_names=None, verbose=True, directions=False, save_csv=False):
        self.true_targets = true_targets
        self.predictions = predictions
        self.pred_proba = pred_proba
        self.output_folder = output_folder
        self.class_names = class_names

        if sp.issparse(self.predictions):
            self.predictions = np.asarray(self.predictions.todense())[0]

        self.auroc = auroc
        self.fscore = fscore
        self.kappa = kappa
        self.acc = acc
        self.f1 = f1

        self.verbose = verbose
        self.f1s = [0.0]
        self.precs = [0.0]
        self.recalls = [0.0]
        self.accs = [0.0]
        self.aurocs = [0.0]
        self.kappas = [0.0]

        super().__init__(true_targets, predictions, pred_proba, file_name, output_folder, save_class, f1=f1, auroc=auroc,
                 fscore=fscore, kappa=kappa, acc=acc, class_names=class_names, verbose=verbose, save_csv=save_csv)

    def calc_fscore(self):
        self.precs.value[0], self.recalls.value[0], self.f1s.value[0], unused__ = precision_recall_fscore_support(self.true_targets, self.predictions, average="binary")
        self.avg_prec.value = self.precs.value[0]
        self.avg_recall.value = self.recalls.value[0]
        self.avg_f1.value = self.f1s.value[0]

    # Check different averages for auroc
    def calc_auroc(self):
        self.aurocs.value[0] = roc_auc_score(self.true_targets, self.pred_proba)
        self.avg_auroc.value = self.aurocs.value[0]

    def calc_acc(self):
        self.accs.value[0] = accuracy_score(self.true_targets, self.predictions)
        self.avg_acc.value = self.accs.value[0]

    def calc_kappa(self):
        self.kappas.value[0] = cohen_kappa_score(self.true_targets, self.predictions)
        self.avg_kappa.value = self.kappas.value[0]

    def save(self):
        self.save_class.save(self.popo_array)

def average_scores(score_dicts):
    new_dict = {}
    for key, value in score_dicts[0].items():
        print(key)
        val_array = []
        for i in range(len(score_dicts)):
            val_array.append(score_dicts[i][key])
        new_dict[key] = np.average(val_array, axis=0)
    return new_dict





if __name__ == '__main__':
    print("k")
