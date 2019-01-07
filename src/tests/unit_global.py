from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

true_targets = [0,0,0,1]
predictions = [0,0,0,0]
prob = [0,0,0,0]
prec, recall, fbeta, score = precision_recall_fscore_support(true_targets, predictions, average="binary")
print("prec", prec, "recall", recall, "fbeta", fbeta, "score", score)
roc = roc_auc_score(true_targets, prob)
print("roc", roc)
acc = accuracy_score(true_targets, predictions)
print("acc", acc)
kappa = cohen_kappa_score(true_targets, predictions)
print("kappa", kappa)

# Test output is numpy if its an array
def standardTests(all_methods):
    print("")

arrays = [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
print(np.average(arrays, axis=0))