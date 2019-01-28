import numpy as np
from util import proj as dt
import pydotplus as pydot
import math
from sklearn import tree
import jsbeautifier
import random
import graphviz
import os
from util import py

from score import classify



class DecisionTree:
    clfs = None
    x_train = None
    y_train = None
    x_test = None
    criterion = None
    max_depth = None
    class_weight = None

    def __init__(self, x_train=None, y_train=None, x_test=None, y_test=None, criterion="entropy", max_depth=-1,
                 class_weight="balanced"):

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.clfs = np.empty(len(y_train), dtype=np.object)
        self.criterion = criterion
        self.max_depth = max_depth
        self.class_weight = class_weight

        self.check_inputs()


    def main(self):
        predictions = self.get_predictions()
        return predictions

    def get_param_dict(self):
        return {"criterion": self.criterion, "max_depth": self.max_depth, "class_weight": self.class_weight}

    def check_inputs(self):
        if not py.isStr(self.criterion) or self.criterion == "balanced" or self.criterion == None:
            raise ValueError("Criterion is the first parameter required, this was entered:", self.criterion)
        if not py.isInt(self.max_depth) and self.max_depth != None:
            raise ValueError("Max_depth is the second parameter required, this was entered:", self.max_depth)
        if not py.isStr(self.class_weight) or self.max_depth == "entropy" or self.max_depth == "gini":
            raise ValueError("Class_weight is the third parameter required, this was entered:", self.class_weight)

    def get_output_names(self):
        return ["predictions", "score_df"]

    def get_predictions(self):
        predictions = np.empty(len(self.y_train), dtype=np.object)
        for i in range(len(self.y_train)):
            self.clfs[i] = tree.DecisionTreeClassifier(max_depth=self.max_depth, criterion=self.criterion,
                                                       class_weight=self.class_weight)
            self.clfs[i].fit(self.x_train, self.y_train[i])
            predictions[i] = self.clfs[i].predict(self.x_test)
            print("Completed class", i, "/", len(self.y_train))
        return predictions

   # Get all of the nodes to a particular depth
    def getNodesToDepth(self, dimension_reps, dimension_names):
        all_clusters = []
        all_rankings = []
        all_fns =[]
        all_ids = []
        for i in range(len(self.clfs)):
            fns = []
            rankings = np.asarray(self.x_train).transpose()
            clusters = np.asarray(dimension_reps)
            features = []
            dt_clusters = []
            for j in range(len(self.clfs[i].tree_.feature)):
                if j != -2 or j <= len(clusters):
                    id = self.clfs[i].tree_.feature[j]
                    if id >=0:
                        fns.append(dimension_names[id])
                        features.append(rankings[id])
                        dt_clusters.append(clusters[id])
            if len(fns) != 1:
                fn_test = np.unique(["".join(map(str, i)) for i in fns], return_index=True)
                fn_ids = fn_test[1]
            else:
                fn_ids = [0]
            final_fns = []
            clusters = list(clusters)
            final_rankings = []
            final_clusters = []
            for i in fn_ids:
                final_fns.append(fns[i])
                final_rankings.append(features[i])
                final_clusters.append(dt_clusters[i])
            all_clusters.append(final_clusters)
            all_rankings.append(final_rankings)
            all_fns.append(final_fns)
            all_ids.append(fn_ids)
        return all_clusters, all_rankings, all_fns, all_ids

    def modifyTree(self, amount_to_modify=3, size=15):
        modified_tree = None
        new_scores = None
        return modified_tree, new_scores

class Visualize:
    clfs = None

    def __init__(self, clfs=None):
        self.clfs = clfs

    def rewrite_tree_dot(self, tree_dot):
        word_dot_data = []
        max = 3
        min = -3
        print(max)
        print(min)
        boundary = max - min
        boundary = boundary / 5
        bound_1 = 0 - boundary * 2
        bound_2 = 0 - boundary * 1
        bound_3 = 0
        bound_4 = 0 + boundary
        bound_5 = 0 + boundary * 2
        for s in tree_dot:
            if ":" in s:
                s = s.split("<=")
                no_num = s[0]
                num = s[1]
                num = num.split()
                end = " ".join(num[:-1])
                num_split = num[0].split("\\")
                num = num_split[0]
                end = end[len(num):]
                num = float(num)
                replacement = ""
                if num <= bound_2:
                    replacement = "VERY LOW"
                elif num <= bound_3:
                    replacement = "VERY LOW - LOW"
                elif num <= bound_4:
                    replacement = "VERY LOW - AVERAGE"
                elif num <= bound_5:
                    replacement = "VERY LOW - HIGH"
                elif num >= bound_5:
                    replacement = "VERY HIGH"
                new_string_a = [no_num, replacement, end]
                new_string = " ".join(new_string_a)
                word_dot_data.append(new_string)
                if "]" in new_string:
                    if '"' not in new_string[len(new_string) - 10:]:
                        for c in range(len(new_string)):
                            if new_string[c + 1] == "]":
                                new_string = new_string[:c] + '"' + new_string[c:]
                                break
        return word_dot_data


    # Returns the tree as a dot file
    def writeTreeDiagrams(self, feature_names, class_names):
        diagrams = []
        for i in range(len(self.clfs)):
            output_names = []
            for c in feature_names:
                line = ""
                counter = 0
                for i in range(len(c)):
                    line = line + c[i] + " "
                    counter += 1
                    if counter == 8:
                        break
                output_names.append(line)
            diagrams.append(tree.export_graphviz(self.clfs[i], feature_names=output_names, class_names=class_names,
                                 label='all', filled=True, impurity=True, node_ids=True,
                                 proportion=True, rounded=True, ))
        return diagrams





    def get_code(self, tree, feature_names, class_names, filename, data_type):
        left      = tree.tree_.children_left
        right     = tree.tree_.children_right
        threshold = tree.tree_.threshold
        value = tree.tree_.value

        features = []
        for i in tree.tree_.feature:
            if i != -2 or i <= 200:
                features.append(feature_names[i])
        rules_array = []
        def recurse(left, right, threshold, features,  node):
                if (threshold[node] != -2):
                        line = "IF ( " + features[node] + " <= " + str(threshold[node]) + " ) {"
                        rules_array.append(line)
                        if left[node] != -1:
                                recurse (left, right, threshold, features,left[node])
                        line = "} ELSE {"
                        rules_array.append(line)
                        if right[node] != -1:
                                recurse (left, right, threshold, features,right[node])
                        line = "}"
                        rules_array.append(line)
                else:
                        if value[node][0][0] >= value[node][0][1]:
                            line = "return", class_names[0]
                            rules_array.append(line)
                        else:
                            line = "return", class_names[1]
                            rules_array.append(line)
        recurse(left, right, threshold, features, 0)
        return jsbeautifier.beautify(rules_array)


def main():
    pass


if  __name__ =='__main__':main()