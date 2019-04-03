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

import pydotplus as pydot
from sklearn import tree

class DecisionTree(Method.ModelMethod):
    max_features = None
    class_weight = None
    verbose = None

    max_depth = None
    min_samples_leaf = None
    min_samples_split = None
    tree_image = None
    feature_names = None
    class_names = None
    tree_image_fn = None

    def __init__(self, x_train, y_train, x_test, y_test, file_name, save_class, probability=True,  max_depth =None,
                                         min_samples_leaf=1, min_samples_split=2, class_weight=None, max_features=None, verbose=False, mcm=None,
                 feature_names=None, class_names=None, get_tree_image=None, tree_image_fn=None):
        self.max_features = max_features
        self.verbose = verbose
        self.class_weight = class_weight
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split=min_samples_split
        self.feature_names = feature_names
        self.class_names = class_names
        self.get_tree_image = get_tree_image
        self.tree_image_fn = tree_image_fn
        super().__init__(x_train, y_train, x_test, y_test, file_name, save_class, probability, mcm)

    def makePopoArray(self):
        super().makePopoArray()
        if self.get_tree_image:
            self.popo_array.append(self.tree_image)

    def makePopos(self):
        super().makePopos()
        self.tree_image = SaveLoadPOPO(self.tree_image, self.tree_image_fn + self.class_names[0] + "dot_data.npy", "npy")

    def process(self):
        # Before we didnt save dictionaries with their types so this is necessary to convert those dicts
        if self.max_features == "None":
            self.max_features = None
        if self.max_depth == "None":
            self.max_depth = None
        elif py.isStr(self.max_depth):
            self.max_depth = int(self.max_depth)
        tree_model = DecisionTreeClassifier(max_features = self.max_features, max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                                     min_samples_split=self.min_samples_split, class_weight=self.class_weight)
        ovr = self.mcm(tree_model)
        ovr.fit(self.x_train, self.y_train)
        self.test_predictions.value = ovr.predict(self.x_test)
        if self.probability:
            self.test_proba.value = ovr.predict_proba(self.x_test)
        if self.get_tree_image:
            self.tree_image.value = []
            for i in range(len(ovr.estimators_)):
                if len(self.class_names) == 1:
                    self.class_names = ["NOT " + self.class_names[0], self.class_names[0]]
                dotfile = tree.export_graphviz(ovr.estimators_[i], out_file=self.tree_image_fn + self.class_names[i] + ".dot", feature_names=self.feature_names, class_names=self.class_names,
                             label='all', filled=True, impurity=True, node_ids=True,
                             proportion=True, rounded=True )
                self.tree_image.value.append(dotfile)
                orig_graph = pydot.graph_from_dot_file(self.tree_image_fn + self.class_names[i] + ".dot")
                orig_graph.write_png(self.tree_image_fn + self.class_names[i] + ".png")
        super().process()

