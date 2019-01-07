from common import Method
from common.SaveLoadPOPO import SaveLoadPOPO
#http://www.jmlr.org/papers/volume18/16-174/16-174.pdf
# This paper states that a 2-fold procedure for resampling/cross-validation is the best for datasets with 1000 or more datapoints
# 2-fold cross validation is where the data-set is split into two, and the model is trained twice, once with the training set being
# The first "fold" and test set being the second fold, and the second where the training set is the second fold.
# Tune C parameters for linear, gammas and C parameters for rbf
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

class RandomForest(Method.ModelMethod):
    max_features = None
    class_weight = None
    verbose = None
    max_features = None

    bootstrap = None
    max_depth = None
    min_samples_leaf = None
    min_samples_split = None

    # C is default 1.0 in the sklearn library, class_weight is balanced as that is the most common for this project
    def __init__(self, x_train, y_train, x_test, y_test, file_name, save_class, probability=True, n_estimators=200,  bootstrap = None, max_depth =None,
                                         min_samples_leaf=None, min_samples_split=None, class_weight=None, max_features=None, verbose=False):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.verbose = verbose
        self.class_weight = class_weight
        self.bootstrap = bootstrap
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split=min_samples_split
        super().__init__(x_train, y_train, x_test, y_test, file_name, save_class, probability)

    def process(self):
        clf = RandomForestClassifier(n_estimators = self.n_estimators, max_features = self.max_features, bootstrap=self.bootstrap, max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                                     min_samples_split=self.min_samples_split, class_weight=self.class_weight)
        clf.fit(self.x_train, self.y_train)
        if self.probability:
            self.test_proba.value = clf.predict_proba(self.x_test)
            self.test_predictions.value = clf.predict(self.x_test)
        else:
            self.test_predictions.value = clf.predict(self.x_test)
        super().process()
