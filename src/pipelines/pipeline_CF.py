import util.classify
import util.text_utils
from util import io as dt
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from keras.datasets import imdb
from rep import pca, ppmi, awv
# import nltk
# nltk.download()
from data import process_corpus
from util.save_load import SaveLoad
from util import split
from pipelines.KFoldHyperParameter import HParam, RecHParam
import os
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier, OutputCodeClassifier
import util.classify
import util.text_utils
from util import io as dt
import numpy as np
#import nltk
#nltk.download()
from data import process_corpus
from util.save_load import SaveLoad
from util import split
import os
from util.text_utils import LimitWords, LimitWordsNumeric
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier, OutputCodeClassifier
from project.get_directions import GetDirections, GetDirectionsSimple
from score.classify import MultiClassScore
from project.get_rankings import GetRankings, GetRankingsStreamed, GetRankingsStreamedSimple
from project.get_ndcg import GetNDCG, GetNDCGStreamed, GetNDCGStreamedSimple
from rep import pca, ppmi, awv, nmf
import scipy.sparse as sp
from project.get_tsr import GetTopScoringRanks, GetTopScoringRanksStreamed
from project.get_tsd import GetTopScoringDirs, GetTopScoringDirsStreamed
from util import py


# The overarching pipeline to obtain all prerequisite data for the derrac pipeline
# Todo: Better standaradize the saving/loading
def pipeline(URM, tags, words, dct, file_name, output_folder, data_type, rewrite_all=False, processed_folder="", LR=False):

    doc_amt = split.get_doc_amt(data_type)
    """
    pca_save = SaveLoad(rewrite=rewrite_all)
    pca_identifier = "_" + str(200) + "_PCA"
    model_fn = file_name + pca_identifier

    pca_instance = pca.PCA(URM, doc_amt, 200,  model_fn, output_folder + "rep/pca/", pca_save)
    pca_instance.process_and_save()
    space = pca_instance.getRep()
    """

    nmf_save = SaveLoad(rewrite=rewrite_all)
    nmf_identifier = "_" + str(200) + "_NMF"
    model_fn = file_name + nmf_identifier

    nmf_instance =  nmf.NMF(URM, doc_amt, 200,  model_fn, output_folder + "rep/nmf/", nmf_save)
    nmf_instance.process_and_save()
    space = nmf_instance.getRep()

    dir_save = SaveLoad(rewrite=rewrite_all)

    file_name = model_fn
    if LR:
        file_name += "_LR"
    print(LR, "LR")


    dir = GetDirectionsSimple(tags, space, dir_save, file_name , processed_folder + "directions/",LR=LR)
    dir.process_and_save()
    # Get rankings on directions save all of them in a word:ranking on entities format, and retrieve if already saved
    dirs = dir.getDirections()

    rank_save = SaveLoad(rewrite=rewrite_all)

    rank = GetRankingsStreamedSimple(dirs, space,  rank_save,  file_name, processed_folder)

    rank.process_and_save()
    rankings = rank.getRankings()

    preds = dir.getPreds()
    score_save = SaveLoad(rewrite=rewrite_all)
    score = MultiClassScore(tags, preds, None, file_name , processed_folder + "directions/score/", score_save, f1=True, auroc=False,
                    fscore=True, kappa=True, acc=True, class_names=words, verbose=False, directions=True, save_csv=True)
    score.process_and_save()
    s_dict = score.get()

    # Dont have to get top dirs cause using tags... (gonna hand pick etc, different process.)


import util.io as dt

def main(data_type, raw_folder, processed_folder, proj_folder="", grams=0, model_type="LinearSVMNOvR", no_below=0.001,
         no_above=0.95, classes_freq_cutoff=100, bowmin=2, dev_percent=0.2, score_metric="avg_f1", max_depth=None,
         multiclass="MOP", LR=True):
    corpus_fn = ""
    if data_type == "animecf":
        URM = sp.load_npz('E:/business/baseline-recommendation\Data_manager_split_datasets\Anime\original/URM_all.npz').transpose()
        import json
        # parse file
        words = np.load("E:/business/baseline-recommendation\Data_manager_split_datasets\Anime\myanimelist/genre_names.npy")
        tags = None#np.asarray(URM.copy().transpose().toarray(), dtype=np.int32)
        words = list(range(URM.shape[0]))
        for i in range(len(words)):
            words[i] = str(words[i])
        dct = None
    if data_type == "mafiascum":
        URM = sp.load_npz(processed_folder + "/bow/num_stw_sparse_corpus.npz")
        tags = sp.load_npz(processed_folder + "/bow/num_stw_sparse_corpus.npz")
        dct = np.load(processed_folder + "/bow/metadata/num_stw_bowdict.pkl")
        words = list(np.load(processed_folder + "/bow/metadata/num_stw_bowdict.pkl").token2id.keys())
        tags = tags.toarray()
    if tags is not None:
        tags[tags >= 1] = 1

        # Run a pipeline that retains numbers and removes stopwords
        ids_to_remove = []
        for i in range(len(tags)):
            if tags[i].max() < 1:
                ids_to_remove.append(i)

        tags = np.delete(tags, ids_to_remove, axis=0)

    multi_class_method = None
    if multiclass == "MOP":
        multi_class_method = MultiOutputClassifier
    elif multiclass == "OVR":
        multi_class_method = OneVsRestClassifier
    elif multiclass == "OVO":
        multi_class_method = OneVsOneClassifier
    elif multiclass == "OCC":
        multi_class_method = OutputCodeClassifier

    pipeline_fn = "num_stw"

    classifier_fn = pipeline_fn + "_" + multiclass
    pipeline( URM, tags, words, dct, pipeline_fn, processed_folder, data_type, rewrite_all=True, processed_folder=processed_folder, LR=LR)


"""
fifty = dt.import2dArray("../../data/processed/placetypes/rep/mds/num_stw_50_MDS.txt")
hundy = dt.import2dArray("../../data/processed/placetypes/rep/mds/num_stw_100_MDS.txt")
two_hundy = dt.import2dArray("../../data/processed/placetypes/rep/mds/num_stw_200_MDS.txt")
np.save("../../data/processed/placetypes/rep/mds/num_stw_50_MDS.npy", fifty)
np.save("../../data/processed/placetypes/rep/mds/num_stw_100_MDS.npy", hundy)
np.save("../../data/processed/placetypes/rep/mds/num_stw_200_MDS.npy", two_hundy)s
"""
# opencyc = np.load("D:\PhD\Code\ThesisPipeline\ThesisPipeline\data_request\Lucas email 1\data\classes/num_stwOpenCYC_classes.npy")
# mds = dt.import2dArray("../../data/processed/placetypes/rep/mds/num_stw_200_MDS.txt")
# np.save("../../data/processed/placetypes/rep/mds/num_stw_200_MDS.npy", mds)

# x = np.load("D:\PhD\Code\ThesisPipeline\ThesisPipeline\data\processed/newsgroups\corpus/num_stw_corpus_processed.npy")
# import scipy.sparse as sp
# xy = sp.load_npz("D:\PhD\Code\ThesisPipeline\ThesisPipeline\data\processed/newsgroups/bow/NB_18_NA_0.95num_stw_ppmi.npz")
if __name__ == '__main__':
    LR = True
    max_depths = [None, None, 3, 2, 1]
    classifiers = ["LinearSVM", "DecisionTree3"]
    data_type = ["animecf"]
    if __name__ == '__main__':
        for j in range(len(data_type)):
            for i in range(len(classifiers)):
                print(data_type[j])
                main(data_type[j], "../../data/raw/" + data_type[j] + "/", "../../data/processed/" + data_type[j] + "/",
                     proj_folder="../../data/proj/" + data_type[j] + "/",
                     grams=0, model_type=classifiers[i], no_below=0.001, no_above=0.95, classes_freq_cutoff=100,
                     bowmin=2, dev_percent=0.2,
                     score_metric="avg_f1", max_depth=max_depths[i], multiclass="OVR", LR=LR)