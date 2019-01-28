from common import Method
from common.SaveLoadPOPO import SaveLoadPOPO
#http://www.jmlr.org/papers/volume18/16-174/16-174.pdf
# This paper states that a 2-fold procedure for resampling/cross-validation is the best for datasets with 1000 or more datapoints
# 2-fold cross validation is where the data-set is split into two, and the model is trained twice, once with the training set being
# The first "fold" and test set being the second fold, and the second where the training set is the second fold.
# Tune C parameters for linear, gammas and C parameters for rbf
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from sklearn.svm import SVC
from util import py
from util.save_load import SaveLoad
from score.classify import MultiClassScore
class SVM(Method.ModelMethod):

    class_weight = None
    verbose = None
    svm = None
    mcm = None

    # C is default 1.0 in the sklearn library, class_weight is balanced as that is the most common for this project
    def __init__(self, x_train, y_train, x_test, y_test, file_name, save_class, probability=False,  class_weight="balanced", verbose=False, mcm=None):
        self.class_weight = class_weight
        self.verbose = verbose
        super().__init__(x_train, y_train, x_test, y_test, file_name, save_class, probability, mcm)


    def process(self):
        if self.probability:
            clf = self.mcm(CalibratedClassifierCV(self.svm))
            clf.fit(self.x_train, self.y_train)
            self.test_proba.value = clf.predict_proba(self.x_test)
            self.test_predictions.value = clf.predict(self.x_test)
        else:
            ovr = self.mcm(self.svm)
            ovr.fit(self.x_train, self.y_train)
            self.test_predictions.value = ovr.predict(self.x_test)

        super().process()


# Get SVM predictions based on one data split/parameter combination, and save them
class LinearSVM(SVM):

    C = None
    class_weight = None
    verbose = None

    # C is default 1.0 in the sklearn library, class_weight is balanced as that is the most common for this project
    def __init__(self, x_train, y_train, x_test, y_test, file_name, save_class, C=1.0, probability=False,  class_weight="balanced", verbose=False, mcm=None):
        self.C = C
        self.class_weight = class_weight
        self.verbose = verbose

        super().__init__(x_train, y_train, x_test, y_test, file_name, save_class, probability,  class_weight, verbose, mcm)

    def process(self):
        print("Begin processing")
        # For old dicts without types
        if self.class_weight == "None":
            self.class_weight = None
        # For some reason, sklearn uses the dual formulation by default for linear SVM's.
        self.svm = LinearSVC(C = float(self.C), class_weight=self.class_weight, dual=False, verbose=self.verbose)
        super().process()



from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier


import numpy as np
from util import split

# Hyper-parameter tuning
# 2-fold cross-validation
# Getting the SVM predictions based on one fold

# SVM just for getting directions
class DirectionSVM(SVM):

    class_weight = None
    verbose = None

    # C is default 1.0 in the sklearn library, class_weight is balanced as that is the most common for this project
    def __init__(self, x_train, y_train, x_test, y_test, file_name, save_class, class_weight="balanced", verbose=False, mcm=None):
        self.class_weight = class_weight
        self.verbose = verbose
        # Probability filled in as false as we are getting directions
        super().__init__(x_train, y_train, x_test, y_test, file_name, save_class, False, class_weight, verbose, mcm)

    def process(self):
        print("Begin processing")
        # For old dicts without types
        if self.class_weight == "None":
            self.class_weight = None
        # For some reason, sklearn uses the dual formulation by default for linear SVM's.
        self.svm = LinearSVC(class_weight="balanced", dual=False, verbose=self.verbose)
        super().process()

# RBF kernel SVM for getting predictions
class GaussianSVM(Method.ModelMethod):

    C = None
    gamma = None
    class_weight = None
    file_name = None
    verbose = None
    probability = None

    # C is default 1.0 in the sklearn library, class_weight is balanced as that is the most common for this project
    def __init__(self, x_train, y_train, x_test, y_test, file_name, save_class, gamma='auto', C=1.0, probability=False, class_weight="balanced",  verbose=False, mcm=None):
        self.C = C
        self.gamma = gamma
        self.file_name = file_name
        self.class_weight = class_weight
        self.verbose = verbose
        self.probability = probability
        super().__init__(x_train, y_train, x_test, y_test, file_name, save_class, C, mcm)

    def process(self):
        print("Begin processing")
        # For some reason, sklearn uses the dual formulation by default for linear SVM's.
        self.svm = SVC(kernel = 'rbf', shrinking=True,  gamma = self.gamma, C = self.C, probability=self.probability, class_weight=self.class_weight, verbose=self.verbose)
        if self.probability:
            clf = CalibratedClassifierCV(self.svm)
            clf.fit(self.x_train, self.y_train)
            self.test_proba.value = clf.predict_proba(self.x_test)
            self.test_predictions.value = clf.predict(self.x_test)
        else:
            ovr = self.svm
            ovr.fit(self.x_train, self.y_train)
            self.test_predictions.value = ovr.predict(self.x_test)
        super().process()
def testSVM(x_train, y_train, x_test, y_test, file_name, class_weight = None, mcm=OneVsRestClassifier):
    svm = testLinearSVM(x_train, y_train, x_test, y_test, file_name, SaveLoad(rewrite=True), class_weight=class_weight, manual_test=True, mcm=mcm)
    svm.process_and_save()
    pred1 = svm.getPred()
    svm2 = testLinearSVM(x_train, y_train, x_test, y_test, file_name, SaveLoad(rewrite=True), class_weight=class_weight,
                    manual_test=False, mcm=mcm)
    svm2.process_and_save()
    pred2 = svm2.getPred()
    return pred1, pred2
class testLinearSVM(SVM):

    C = None
    class_weight = None
    verbose = None
    manuel_test = None

    # C is default 1.0 in the sklearn library, class_weight is balanced as that is the most common for this project
    def __init__(self, x_train, y_train, x_test, y_test, file_name, save_class, C=1.0, probability=False,  class_weight="balanced", verbose=False, mcm=None, manual_test=False):
        self.C = C
        self.class_weight = class_weight
        self.verbose = verbose

        self.manual_test = manual_test

        super().__init__(x_train, y_train, x_test, y_test, file_name, save_class, probability,  class_weight, verbose, mcm)

    def process(self):
        print("Begin processing")
        # For old dicts without types
        if self.class_weight == "None":
            self.class_weight = None
        # For some reason, sklearn uses the dual formulation by default for linear SVM's.
        self.svm = LinearSVC(C = float(self.C), class_weight=self.class_weight, dual=False, verbose=self.verbose)
        if self.manual_test:
            self.y_test = py.transIfRowsLarger(self.y_test)
            self.y_train = py.transIfRowsLarger(self.y_train)
            self.test_predictions.value = []
            for i in range(len(self.y_test)):
                ovr = self.svm
                ovr.fit(self.x_train, self.y_train[i])
                self.test_predictions.value.append(ovr.predict(self.x_test))
            self.test_predictions.value = np.asarray(self.test_predictions.value).transpose()
        else:
            super().process()
if __name__ == '__main__':
    file_name = "test_run_xo"
    data_type = "reuters"
    orig_fn = "../../data/processed/" + data_type + "/"
    corpus = np.load(orig_fn + "rep/pca/num_stw_50_PCA.npy")
    classes = np.load(orig_fn + "classes/num_stw_classes.npy")

    split_ids = split.get_split_ids(data_type, None)
    x_train, y_train, x_test, y_test, x_dev, y_dev = split.split_data(corpus, classes, split_ids)


    pred1, pred2 = testSVM(x_train, y_train, x_dev, y_dev, file_name)
    score = MultiClassScore(y_dev, pred1, None, file_name, orig_fn + "rep/score/", SaveLoad(rewrite=True))
    score.process_and_save()
    scores = score.get()

    score = MultiClassScore(y_dev, pred2, None, file_name, orig_fn + "rep/score/", SaveLoad(rewrite=True))
    score.process_and_save()
    scores2 = score.get()
