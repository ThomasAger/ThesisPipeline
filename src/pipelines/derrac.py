# Save/load dependent on parameters
# When classifying, e.g. with tree, limit entities according to those in the classification problem
# Cross-validation on trees
# Filename manager?
from file_io import csv
from model import tree
from score import classify
from util import split, filename as fnm


def pipeline(data_type=None, all_x=None, all_y=None, split_ids=None, property_names=None,
              min_freq=10, max_freq=100, score_type="acc", epochs=100, cluster_type="derrac", word_vectors=0.0,
             cluster_centers=2, cluster_center_directions=400, cluster_directions=2000, bandwidth=1, n_init=10, max_iter=300):


    # If not using cross-validation, then get the data-splits according to pre-defined standards for each dataset
    if split_ids is None:
        split_ids = split.get_split_ids(data_type)

    # Split the initial data
    file_name = fnm.build_filename(previous_fn=data_type, name_dict=split.get_name_dict(method_name=split, params=[all_x, all_y, split_ids]))
    x_train, y_train, x_test, y_test = split.split_data(all_x, all_y, split_ids)

    # Initialize a tree
    d3_tree = tree.DecisionTree(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                                criterion="entropy",  max_depth=3, class_weight="balanced")

    # Train the tree and get predictions
    predictions = d3_tree.get_predictions()

    # Initialize a classifier scoring object
    d3_score = classify.MultiClassScore(predictions, y_test, auroc=True, fscore=True, kappa=True, acc=True)
    # Calculate all the score types and return the auroc score
    score_dict = d3_score.get()
    csv.save_multiclass_score(score_dict, file_name)


    print("d3 auroc average", d3_average_auroc, d3_score)