from common import Method
from common.SaveLoadPOPO import SaveLoadPOPO
#http://www.jmlr.org/papers/volume18/16-174/16-174.pdf
# This paper states that a 2-fold procedure for resampling/cross-validation is the best for datasets with 1000 or more datapoints
# 2-fold cross validation is where the data-set is split into two, and the model is trained twice, once with the training set being
# The first "fold" and test set being the second fold, and the second where the training set is the second fold.
# Tune C parameters for linear, gammas and C parameters for rbf
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

class SVM(Method.ModelMethod):

    C = None
    class_weight = None
    verbose = None
    svm = None

    # C is default 1.0 in the sklearn library, class_weight is balanced as that is the most common for this project
    def __init__(self, x_train, y_train, x_test, y_test, file_name, save_class, C=1.0, probability=False,  class_weight="balanced", verbose=False):
        self.C = C
        self.class_weight = class_weight
        self.verbose = verbose
        super().__init__(x_train, y_train, x_test, y_test, file_name, save_class, probability)

    def process(self):
        if self.probability:
            clf = OneVsRestClassifier(CalibratedClassifierCV(self.svm))
            clf.fit(self.x_train, self.y_train)
            self.test_proba.value = clf.predict_proba(self.x_test)
            self.test_predictions.value = clf.predict(self.x_test)
        else:
            ovr = OneVsRestClassifier(self.svm)
            ovr.fit(self.x_train, self.y_train)
            self.test_predictions.value = ovr.predict(self.x_test)
        super().process()


# Get SVM predictions based on one data split/parameter combination, and save them
class LinearSVM(SVM):

    C = None
    class_weight = None
    verbose = None

    # C is default 1.0 in the sklearn library, class_weight is balanced as that is the most common for this project
    def __init__(self, x_train, y_train, x_test, y_test, file_name, save_class, C=1.0, probability=False,  class_weight="balanced", verbose=False):
        self.C = C
        self.class_weight = class_weight
        self.verbose = verbose
        super().__init__(x_train, y_train, x_test, y_test, file_name, save_class, C, probability,  class_weight, verbose)

    def process(self):
        print("Begin processing")
        # For old dicts without types
        if self.class_weight == "None":
            self.class_weight = None
        # For some reason, sklearn uses the dual formulation by default for linear SVM's.
        self.svm = LinearSVC(C = float(self.C), class_weight=self.class_weight, dual=False, verbose=self.verbose)
        super().process()

# Hyper-parameter tuning
# 2-fold cross-validation
# Getting the SVM predictions based on one fold

# SVM just for getting directions
class DirectionSVM(Method.Method):
    def process(self):
        print("Begin processing")
        clf = LinearSVC(C = self.C, class_weight=self.class_weight, verbose=True)
        clf.fit(self.x_train, self.y_train)
        direction = clf.dual_coef_.tolist()[0]
        y_pred = clf.predict(x_test)
        y_pred = y_pred.tolist()
        super().process()
    print("")

# RBF kernel SVM for getting predictions
class GaussianSVM(Method.ModelMethod):

    C = None
    gamma = None
    class_weight = None
    file_name = None
    verbose = None
    probability = None

    # C is default 1.0 in the sklearn library, class_weight is balanced as that is the most common for this project
    def __init__(self, x_train, y_train, x_test, y_test, file_name, save_class, gamma='auto', C=1.0, probability=False, class_weight="balanced",  verbose=False):
        self.C = C
        self.gamma = gamma
        self.file_name = file_name
        self.class_weight = class_weight
        self.verbose = verbose
        self.probability = probability
        super().__init__(x_train, y_train, x_test, y_test, file_name, save_class, C)

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