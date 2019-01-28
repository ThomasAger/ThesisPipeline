from util import io as dt
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from keras.datasets import imdb
from rep import pca, ppmi, awv
#import nltk
#nltk.download()
from data import process_corpus
from util.save_load import SaveLoad
from util import split
from pipelines.KFoldHyperParameter import HParam, RecHParam
import os

# The overarching pipeline to obtain all prerequisite data for the derrac pipeline
# Todo: Better standaradize the saving/loading
def pipeline(file_name, space, bow, classes, class_names, processed_folder, dims, kfold_hpam_dict, hpam_dict,
                     model_type=model_type, dev_percent=dev_percent, rewrite_all=False, remove_stop_words=True,
                     score_metric="avg_f1", auroc=False, dir_min_freq=None, dir_max_freq=None, name_of_class):
    if data_type == "movies" or data_type == "placetypes":
        classifier_fn = file_name + "_" + name_of_class + "_"
    else:
        classifier_fn = file_name

    x_train = None
    y_train = None
    x_test = None
    y_test = None
    x_dev = None
    y_dev = None

    probability = False
    if auroc is True:
        probability = True

    doc_amt = split.get_doc_amt(data_type)

    no_below = int(doc_amt * dir_min_freq)
    no_above = int(doc_amt * dir_max_freq)
    print("Filtering all words that do not appear in", no_below, "documents")
    classes_save = SaveLoad(rewrite=rewrite_all)
    classes_process = process_corpus.getDirections(space, class_names, file_name, processed_folder,no_below,
                                         no_above, remove_stop_words, classes_save, name_of_class)
    classes_process.process_and_save()
    classes = classes_process.getClasses()
    class_names = classes_process.getClassNames()



def main(data_type, raw_folder, processed_folder,proj_folder="",  grams=0, model_type="LinearSVM", dir_min_freq=0.001,
         dir_max_freq=0.95, dev_percent=0.2, score_metric="avg_f1", max_depth=None):
    pipeline_fn = "num_stw"
    if data_type == "newsgroups":
        name_of_class = ["Newsgroups"]
    elif data_type == "sentiment":
        name_of_class = ["Sentiment"]
    elif data_type == "movies":
        name_of_class = ["Genres", "Keywords", "Ratings"]
    elif data_type == "placetypes":
        name_of_class = ["Foursquare", "Geonames", "OpenCYC"]
    elif data_type == "reuters":
        name_of_class = ["Reuters"]

    window_size = [5, 10, 15]
    min_count = [1, 5, 10]
    train_epoch = [50, 100, 200]

    dims = [50, 100, 200]
    balance_params = ["balanced", None]
    C_params = [1.0, 0.01, 0.001, 0.0001]
    gamma_params = [1.0, 0.01, 0.001, 0.0001]

    n_estimators = [ 1000, 2000]
    max_features = [None, 'auto', 'log2']
    criterion = ["gini", "entropy"]
    max_depth = [max_depth]
    bootstrap = [True, False]
    min_samples_leaf = [1]
    min_samples_split = [2]

    # Run a pipeline that retains numbers and removes stopwords

    if model_type == "GaussianSVM" or model_type == "GaussianSVMMultiClass":
        kfold_hpam_dict = {"C":C_params,
                    "class_weight":balance_params,
                           "gamma":gamma_params}
    elif model_type == "LinearSVM":
        kfold_hpam_dict = {"C":C_params,
                    "class_weight":balance_params}
    elif model_type == "RandomForest":
        kfold_hpam_dict = {"n_estimators":n_estimators,
                           "max_features":max_features,
                    "class_weight":balance_params,
                           "criterion":criterion,
                           "max_depth":max_depth,
                           "bootstrap":bootstrap,
                           "min_samples_leaf":min_samples_leaf,
                           "min_samples_split":min_samples_split}
    elif model_type[:12] == "DecisionTree":
        kfold_hpam_dict = {"max_features":max_features,
                    "class_weight":balance_params,
                           "criterion":criterion,
                           "max_depth":max_depth}
    hpam_dict = { "window_size":window_size,
                "min_count":min_count,
                "train_epoch":train_epoch}

    for ci in range(len(name_of_class)):
        classes_save = SaveLoad(rewrite=False)
        # These were the parameters used for the previous experiments
        no_below = 0.0001
        no_above = 0.95
        bowmin = 2
        classes_freq_cutoff = 100
        classes_process = process_corpus.ProcessClasses(None, None, pipeline_fn, processed_folder, bowmin, no_below,
                                                        no_above, classes_freq_cutoff, True, classes_save, name_of_class[ci])
        bow = process_corpus.getBow()
        classes = classes_process.getClasses()
        class_names = classes_process.getClassNames()
        for i in range(len(dims)):
            spaces = []
            space_names = []
            pca_identifier = "_" + str(dims[i]) + "_PCA"
            pca_fn = pipeline_fn + pca_identifier
            pca_instance = pca.PCA(None, None, dims[i],
                                   pca_fn, processed_folder + "rep/pca/", SaveLoad(rewrite=False))
            pca_instance.process_and_save()
            spaces.append(pca_instance.getRep())
            space_names.append(pca_fn)

            awv_identifier = "_" + str(dims[i]) + "_AWVEmp"
            awv_fn = pipeline_fn + awv_identifier
            awv_instance = awv.AWV(None, dims[i], awv_fn, processed_folder + "rep/awv/", SaveLoad(rewrite=False))
            awv_instance.process_and_save()
            spaces.append(awv_instance.getRep())
            space_names.append(awv_fn)

            if data_type != "sentiment":
                mds_identifier = "_" + str(dims[i]) + "_MDS"
                mds_fn = pipeline_fn + mds_identifier
                import_fn = processed_folder + "rep/mds/" + mds_fn + ".npy"
                spaces.append(dt.import2dArray(import_fn))
                space_names.append(mds_fn)

            if data_type != "movies" and data_type != "placetypes":
                doc2vec_identifier = "_" + str(dims[i]) + "_D2V"
                doc2vec_fn = pipeline_fn + doc2vec_identifier
                classifier_fn = pipeline_fn + "_" + name_of_class[ci] + "_"

                wv_path_d2v = os.path.abspath("../../data/raw/glove/" + "glove.6B.300d.txt")

                corpus_fn = processed_folder + "corpus/" + "num_stw_corpus_processed.txt"
                hpam_dict["dim"] = [dims[i]]
                hpam_dict["corpus_fn"] = [corpus_fn]
                hpam_dict["wv_path"] = [wv_path_d2v]

                # Have to leave classes in due to messy method
                hyper_param = RecHParam(None, classes, None, hpam_dict, hpam_dict, "d2v", "LinearSVM",
                                        doc2vec_fn, classifier_fn, processed_folder, SaveLoad(rewrite=False), data_type=data_type,
                                        score_metric=score_metric)
                d2v_space, __unused = hyper_param.getTopScoringSpace()
                spaces.append(d2v_space)
                space_names.append(doc2vec_fn)

            for s in range(len(spaces)):
                pipeline(pipeline_fn, spaces[s], bow, classes, class_names, processed_folder, dims, kfold_hpam_dict, hpam_dict,
                     model_type=model_type, dev_percent=dev_percent, rewrite_all=False, remove_stop_words=True,
                     score_metric=score_metric, auroc=False, dir_min_freq=dir_min_freq, dir_max_freq=dir_max_freq, name_of_class=name_of_class[i])

"""
fifty = dt.import2dArray("../../data/processed/placetypes/rep/mds/num_stw_50_MDS.txt")
hundy = dt.import2dArray("../../data/processed/placetypes/rep/mds/num_stw_100_MDS.txt")
two_hundy = dt.import2dArray("../../data/processed/placetypes/rep/mds/num_stw_200_MDS.txt")
np.save("../../data/processed/placetypes/rep/mds/num_stw_50_MDS.npy", fifty)
np.save("../../data/processed/placetypes/rep/mds/num_stw_100_MDS.npy", hundy)
np.save("../../data/processed/placetypes/rep/mds/num_stw_200_MDS.npy", two_hundy)
"""
mds = dt.import2dArray("../../data/processed/placetypes/rep/mds/num_stw_200_MDS.txt")
np.save("../../data/processed/placetypes/rep/mds/num_stw_200_MDS.npy", mds)
max_depths = [None, None, 3, 2, 1]
classifiers = ["LinearSVM", "DecisionTreeNone", "DecisionTree3", "DecisionTree2", "DecisionTree1"]
data_type = "reuters"
if __name__ == '__main__':
    for i in range(len(classifiers)):
        main(data_type, "../../data/raw/"+data_type+"/",  "../../data/processed/"+data_type+"/", proj_folder="../../data/proj/"+data_type+"/",
                                grams=0, model_type=classifiers[i], dir_min_freq=0.001, dir_max_freq=0.95, dev_percent=0.2,
                                score_metric="avg_f1", max_depth=max_depths[i])