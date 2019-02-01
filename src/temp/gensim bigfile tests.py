
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def import1dArray(file_name, file_type="s"):
    with open(file_name, "r", encoding="cp1252") as infile:
        if file_type == "f":
            array = []
            lines = infile.readlines()
            for line in lines:
                array.append(float(line.strip()))
        elif file_type == "i":
            array = [int(float(line.strip())) for line in infile]
        else:
            array = [line.strip() for line in infile]
    return np.asarray(array)

mds_space = np.load("../data/mds/num_stw_50_MDS.npy")
entity_names = import1dArray("../data/mds/entity_names.txt")
optimal_C = 1.0
class_weights = "balanced"
classes = np.load("../data/classes/num_stwFoursquare_classes.npy")
class_entities = import1dArray("../data/classes/Foursquare_entities.txt")

# Getting the matching mds vectors available for the class
matched_ids = []
for i in range(len(class_entities)):
    for j in range(len(entity_names)):
        if class_entities[i] == entity_names[j]:
            matched_ids.append(j)
            break

# Splitting into test/train
x_train_split = int(len(matched_ids) * 0.66)
y_ids = list(range(len(matched_ids)))
y_train_split = int(len(matched_ids) * 0.66)
y_train = y_ids[:y_train_split]
y_test = y_ids[y_train_split:]
x_train = ids[:x_train_split]
x_test = ids[x_train_split:]

# Removing the dev data used to train the hyper-parameters
x_train = x_train[:int(len(x_train) * (1 - 0.2))]
y_train = y_train[:int(len(y_train) * (1 - 0.2))]

# For some reason, dual is set to True by default
svm = OneVsRestClassifier(LinearSVC(C = optimal_C, class_weight=class_weights, dual=False, verbose=True))

svm.fit(x_train, y_train)
pred = clf.predict_proba(self.x_test)

precs = np.zeros(len(y_train[0]))
recalls = np.zeros(len(y_train[0]))
f1s = np.zeros(len(y_train[0]))

for i in range(len(f1s)):
    precs.value[i], recalls.value[i], f1s.value[i], unused__ = precision_recall_fscore_support(
        true_targets[i], predict, average="binary")

avg_prec = np.average(self.precs.value)
avg_recall = np.average(self.recalls.value)
avg_f1 = get_f1_score(self.avg_prec.value, self.avg_recall.value)

print("prec", avg_prec,"recall", avg_recall, "f1", avg_f1)