
import math
import os.path

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score
from sklearn.metrics import precision_recall_fscore_support

from util import io


def get_f1_score(prec, recall):
    f1 = 2 * ((prec * recall) / (prec + recall))
    if math.isnan(f1):
        f1 = 0.0
    return f1

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

class MultiClassScore():
    
    true_targets = None
    predictions = None

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


    def __init__(self, true_targets, predictions, auroc=False, fscore=False, kappa=False, acc=False, verbose=True):
        self.true_targets = true_targets
        self.predictions = predictions
        self.check()
        self.f1s = np.full(len(predictions), math.nan)
        self.precs = np.full(len(predictions), math.nan)
        self.recalls = np.full(len(predictions), math.nan)
        self.accs = np.full(len(predictions), math.nan)
        self.aurocs = np.full(len(predictions), math.nan)
        self.kappas = np.full(len(predictions), math.nan)
        if auroc:
            self.auroc()
        if fscore:
            self.fscore()
        if kappa:
            self.kappa()
        if acc:
            self.acc()
        if verbose:
            self.print()



    def get(self):
        score_dict = {}
        if not math.isnan(self.f1s[0]):
            score_dict["f1"] = self.f1s
            score_dict["avg_f1"] = self.avg_f1

        if not math.isnan(self.precs[0]):
            score_dict["prec"] = self.precs

        if not math.isnan(self.recalls[0]):
            score_dict["recall"] = self.recalls

        if not math.isnan(self.accs[0]):
            score_dict["acc"] = self.accs
            score_dict["avg_acc"] =  self.avg_acc

        if not math.isnan(self.aurocs[0]):
            score_dict["auroc"] = self.aurocs
            score_dict["avg_auroc"] = self.avg_auroc

        if not math.isnan(self.kappas[0]):
            score_dict["kappa"] = self.kappas
            score_dict["avg_kappa"] = self.avg_kappa
        return score_dict

    def print(self, max_to_print=5):
        score_dict = {}
        print("Amt", len(self.f1s))

        if not math.isnan(self.precs[0]):
            print("prec", self.precs[:max_to_print])

        if not math.isnan(self.recalls[0]):
            print("recall",  self.recalls[:max_to_print])

        if not math.isnan(self.f1s[0]):
            print("f1s",  self.f1s[:max_to_print], "| average", self.avg_f1)

        if not math.isnan(self.accs[0]):
            print("acc", self.accs[:max_to_print], "| average", self.avg_acc)

        if not math.isnan(self.aurocs[0]):
            print("auroc", self.aurocs[:max_to_print], "| average", self.avg_auroc)

        if not math.isnan(self.kappas[0]):
            print("kappa",  self.kappas[:max_to_print], "| average", self.avg_kappa)
        return score_dict

    def fscore(self):
        for i in range(len(self.predictions)):
            self.precs[i], self.recalls[i], self.f1s[i], unused__ = precision_recall_fscore_support(self.true_targets[i], self.predictions[i], average="binary")
        self.avg_f1 = get_f1_score(np.average(self.precs), np.average(self.recalls))

    # Check different averages for auroc
    def auroc(self):
        for i in range(len(self.predictions)):
            self.aurocs[i] = roc_auc_score(self.true_targets[i], self.predictions[i])
        self.avg_auroc = np.average(self.aurocs)

    def acc(self):
        for i in range(len(self.predictions)):
            self.accs[i] = accuracy_score(self.true_targets[i], self.predictions[i])
        self.avg_acc = np.average(self.accs)

    def kappa(self):
        for i in range(len(self.predictions)):
            self.kappas[i] = cohen_kappa_score(self.true_targets[i], self.predictions[i])
        self.avg_kappa = np.average(self.kappas)

    def save_csv(self, csv_fn):

        csv_acc = self.all_acc
        csv_f1 = self.all_f1
        csv_auroc = self.all_auroc

        csv_acc.append(self.average_acc)
        csv_f1.append(self.average_f1)
        csv_auroc.append(self.average_auroc)

        scores = [csv_acc, csv_f1, csv_auroc]
        col_names = ["acc", "f1", "auroc"]
        if os.path.exists(csv_fn):
            print("File exists, writing to csv")
            try:
                dt.write_to_csv(csv_fn, col_names, scores)
                return
            except PermissionError:
                print("CSV FILE WAS OPEN, SKIPPING")
                return
            except ValueError:
                print("File does not exist, recreating csv")
        print("File does not exist, recreating csv")
        key = []
        for l in self.class_names:
            key.append(l)
        key.append("AVERAGE")
        key.append("MICRO AVERAGE")
        io.write_csv(csv_fn, col_names, scores, key)

if __name__ == '__main__':
    r_true_targets = np.random.randint(2,size=(11, 40),dtype=np.int8)
    r_predictions = np.random.randint(2,size=(11, 40),dtype=np.int8)
    score = MultiClassScore(r_true_targets, r_predictions)
    score.auroc()
    score.fscore()
    score.acc()
    score.kappa()
    score.print()
    results = score.get()
    print("--------------------------------------")
    alternate_results = MultiClassScore(r_true_targets, r_predictions, auroc=True, fscore=True, acc=True, kappa=True, verbose=True).get()
    new_score = MultiClassScore(r_true_targets, r_predictions, auroc=True, fscore=True, acc=True, kappa=True,
                                        verbose=True)
    alternate_results_2 = new_score.get()
