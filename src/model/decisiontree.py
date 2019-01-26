from common import Method
from common.SaveLoadPOPO import SaveLoadPOPO
#http://www.jmlr.org/papers/volume18/16-174/16-174.pdf
# This paper states that a 2-fold procedure for resampling/cross-validation is the best for datasets with 1000 or more datapoints
# 2-fold cross validation is where the data-set is split into two, and the model is trained twice, once with the training set being
# The first "fold" and test set being the second fold, and the second where the training set is the second fold.
# Tune C parameters for linear, gammas and C parameters for rbf
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from util import py

class DecisionTree(Method.ModelMethod):
    max_features = None
    class_weight = None
    verbose = None

    max_depth = None
    min_samples_leaf = None
    min_samples_split = None

    def __init__(self, x_train, y_train, x_test, y_test, file_name, save_class, probability=True,  max_depth =None,
                                         min_samples_leaf=1, min_samples_split=2, class_weight=None, max_features=None, verbose=False):
        self.max_features = max_features
        self.verbose = verbose
        self.class_weight = class_weight
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split=min_samples_split
        super().__init__(x_train, y_train, x_test, y_test, file_name, save_class, probability)


    def process(self):
        # Before we didnt save dictionaries with their types so this is necessary to convert those dicts
        if self.max_features == "None":
            self.max_features = None
        if self.max_depth == "None":
            self.max_depth = None
        elif py.isStr(self.max_depth):
            self.max_depth = int(self.max_depth)
        tree = DecisionTreeClassifier(max_features = self.max_features, max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                                     min_samples_split=self.min_samples_split, class_weight=self.class_weight)
        ovr = OneVsRestClassifier(tree)
        ovr.fit(self.x_train, self.y_train)
        self.test_predictions.value = ovr.predict(self.x_test)
        if self.probability:
            self.test_proba.value = ovr.predict_proba(self.x_test)
        super().process()