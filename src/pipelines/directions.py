from util import io as dt
import numpy as np
from rep import pca, awv
#import nltk
#nltk.download()
from data import process_corpus
from util.save_load import SaveLoad
from util import split
from pipelines.KFoldHyperParameter import RecHParam
import os
from data.process_corpus import LimitWords
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier, OutputCodeClassifier
from project.get_directions import GetDirections
from score.classify import MultiClassScore
from project.get_rankings import GetRankings
from project.get_ndcg import GetNDCG
from rep import pca, ppmi, awv
from sklearn import linear_model

# The overarching pipeline to obtain all prerequisite data for the derrac pipeline
# Todo: Better standaradize the saving/loading
last_dct = []
def pipeline(file_name, space, bow, dct, classes, class_names, words_to_get, processed_folder, dims, kfold_hpam_dict, hpam_dict,
                     model_type="", dev_percent=0.2, rewrite_all=False, remove_stop_words=True,
                     score_metric="", auroc=False, dir_min_freq=0.001, dir_max_freq=0.95, name_of_class="",
             classifier_fn="", mcm=None, top_scoring_dirs=2000, score_type="kappa", ppmi=None, dct_unchanged=None):

    # Convert to hyper-parameter method, where hyper-parameters are:
    ##### dir_min_freq, dir_max freq
    ##### Scoring filters, aka top 200, top 400, etc, score-type
    dct_len_start = len(dct_unchanged.dfs.keys())
    if bow.shape[0] != len(dct.dfs.keys()) or len(dct.dfs.keys()) != ppmi.shape[0]:
        print("bow", bow.shape[0], "dct", len(dct.dfs.keys()), "ppmi", ppmi.shape[0])
        raise ValueError("Size of vocab and dict do not match")


    doc_amt = split.get_doc_amt(data_type)

    no_below = int(doc_amt * dir_min_freq)
    no_above = int(doc_amt * dir_max_freq)



    print("(For directions) Filtering all words that do not appear in", no_below, "documents")

    wl_save = SaveLoad(rewrite=rewrite_all)

    dir = LimitWords(file_name, wl_save, dct, bow, processed_folder +"directions/words/", words_to_get, no_below, no_above)
    dir.process_and_save()
    words_to_get = dir.getBowWordDct()
    new_word_dict = dir.getNewWordDict()


    # Rewrite is always true for this as loading is handled internally
    dir_save = SaveLoad(rewrite=rewrite_all)
    dir = GetDirections(bow, space, words_to_get, new_word_dict, dir_save, no_below, no_above, file_name , processed_folder + "directions/", LR=False)
    dir.process_and_save()
    binary_bow = np.asarray(dir.getNewBow().todense(), dtype=np.int32)
    binary_bow[binary_bow >= 1] = 1
    preds = dir.getPreds()
    words = dir.getWords()

    score_save = SaveLoad(rewrite=rewrite_all)
    score = MultiClassScore(binary_bow, preds, None, file_name + "_" + str(no_below) + "_" + str(no_above) , processed_folder + "directions/score/", score_save, f1=True, auroc=False,
                    fscore=True, kappa=True, acc=True, class_names=words, verbose=False, directions=True, save_csv=True)
    score.process_and_save()
    score.print()

    # Get rankings on directions save all of them in a word:ranking on entities format, and retrieve if already saved
    dirs = dir.getDirections()

    rank_save = SaveLoad(rewrite=rewrite_all)
    rank = GetRankings(dirs, space, words,  rank_save,  file_name, processed_folder, no_below, no_above)
    rank.process_and_save()
    rankings = rank.getRankings()

    dct_len_end =  len(dct_unchanged.dfs.keys())
    if dct_len_start != dct_len_end:
        raise ValueError("Dct has changed shape")

    # Get NDCG scores
    ndcg_save = SaveLoad(rewrite=True)
    ndcg = GetNDCG(rankings, ppmi, words, dct_unchanged.token2id,  ndcg_save,  file_name, processed_folder + "rank/ndcg/", no_below, no_above)
    ndcg.process_and_save()
    ndcg_scores = ndcg.getNDCG()

    # Filter directions based on the amount to filter in params and the score type to filter in params

    # Convert to entity representation from words

    # Score rankings on directions (SVM, Decision tree, etc)






def main(data_type, raw_folder, processed_folder,proj_folder="",  grams=0, model_type="LinearSVM", dir_min_freq=0.001,
         dir_max_freq=0.95, dev_percent=0.2, score_metric="avg_f1", max_depth=None, multiclass="OVR", LR=False, bonus_fn="",
         rewrite_all = False):
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

    multi_class_method = None
    if multiclass == "MOP":
        multi_class_method = MultiOutputClassifier
    elif multiclass == "OVR":
        multi_class_method = OneVsRestClassifier
    elif multiclass == "OVO":
        multi_class_method = OneVsOneClassifier
    elif multiclass == "OCC":
        multi_class_method = OutputCodeClassifier

    for ci in range(len(name_of_class)):
        classes_save = SaveLoad(rewrite=False)
        # These were the parameters used for the previous experiments
        no_below = 0.0001
        no_above = 0.95
        bowmin = 2
        classes_freq_cutoff = 100
        # The True here and below is to remove stop words
        classes_process = process_corpus.ProcessClasses(None, None, pipeline_fn, processed_folder, bowmin, no_below,
                                                        no_above, classes_freq_cutoff, True, classes_save, name_of_class[ci])

        classes = classes_process.getClasses()
        class_names = classes_process.getClassNames()

        corp_save = SaveLoad(rewrite=False)
        p_corpus = process_corpus.Corpus(None, classes, name_of_class[ci], pipeline_fn, processed_folder,
                                         bowmin,
                                         no_below, no_above, True, corp_save)
        bow = p_corpus.getBow()
        word_list = p_corpus.getAllWords()

        ppmi_save = SaveLoad(rewrite=rewrite_all)
        ppmi_identifier = "_ppmi"
        ppmi_fn = pipeline_fn + ppmi_identifier

        ppmi_unfiltered = ppmi.PPMI(p_corpus.getBow(), None, processed_folder + "bow/" + ppmi_fn, ppmi_save)
        ppmi_unfiltered.process_and_save()
        ppmi_unf_matrix = ppmi_unfiltered.getMatrix().transpose()

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
                if len(bonus_fn) != 0:
                    print("WARNING, bonus fn active")
                if LR:
                    final_fn = pipeline_fn + "_LR_"+ space_names[s]
                else:
                    final_fn = pipeline_fn + "_"+ space_names[s]

                final_fn += bonus_fn

                corp_save = SaveLoad(rewrite=False)
                p_corpus = process_corpus.Corpus(None, classes, name_of_class[ci], pipeline_fn, processed_folder,
                                                 bowmin,
                                                 no_below, no_above, True, corp_save)
                # Reload the dict because gensim is persistent otherwise
                dct = p_corpus.getBowDct()
                dct_unchanged = p_corpus.getBowDct()

                if data_type == "movies" or data_type == "placetypes":
                    classifier_fn = final_fn + "_" + name_of_class[i] + "_" + multiclass
                    pipeline(final_fn, spaces[s], bow, dct, classes[ci], class_names[ci], word_list, processed_folder, dims, kfold_hpam_dict, hpam_dict,
                 model_type=model_type, dev_percent=dev_percent, rewrite_all=rewrite_all, remove_stop_words=True,
                 score_metric=score_metric, auroc=False, dir_min_freq=dir_min_freq, dir_max_freq=dir_max_freq, name_of_class=name_of_class[ci], classifier_fn = classifier_fn,
                             mcm=multi_class_method, ppmi=ppmi_unf_matrix, dct_unchanged=dct_unchanged)
                else:
                    classifier_fn = pipeline_fn + "_" + multiclass
                    pipeline(final_fn, spaces[s], bow, dct, classes, class_names, word_list, processed_folder, dims, kfold_hpam_dict, hpam_dict,
                     model_type=model_type, dev_percent=dev_percent, rewrite_all=rewrite_all, remove_stop_words=True,
                     score_metric=score_metric, auroc=False, dir_min_freq=dir_min_freq, dir_max_freq=dir_max_freq, name_of_class=name_of_class, classifier_fn = classifier_fn,
                             mcm=multi_class_method, ppmi=ppmi_unf_matrix, dct_unchanged=dct_unchanged)


"""
fifty = dt.import2dArray("../../data/processed/placetypes/rep/mds/num_stw_50_MDS.txt")
hundy = dt.import2dArray("../../data/processed/placetypes/rep/mds/num_stw_100_MDS.txt")
two_hundy = dt.import2dArray("../../data/processed/placetypes/rep/mds/num_stw_200_MDS.txt")
np.save("../../data/processed/placetypes/rep/mds/num_stw_50_MDS.npy", fifty)
np.save("../../data/processed/placetypes/rep/mds/num_stw_100_MDS.npy", hundy)
np.save("../../data/processed/placetypes/rep/mds/num_stw_200_MDS.npy", two_hundy)
"""
max_depths = [None, None, 3, 2, 1]
classifiers = ["LinearSVM", "DecisionTreeNone", "DecisionTree3", "DecisionTree2", "DecisionTree1"]
data_type = "movies"
doLR = False
if data_type == "placetypes":
    dminf = 0.05
else:
    dminf = 0.1
multi_class_method = "OVR"
bonus_fn = ""
rewrite_all=False
if __name__ == '__main__':
    for i in range(len(classifiers)):
        main(data_type, "../../data/raw/"+data_type+"/",  "../../data/processed/"+data_type+"/", proj_folder="../../data/proj/"+data_type+"/",
                                grams=0, model_type=classifiers[i], dir_min_freq=dminf, dir_max_freq=0.95, dev_percent=0.2,
                                score_metric="avg_f1", max_depth=max_depths[i], multiclass=multi_class_method, LR=doLR, bonus_fn=bonus_fn, rewrite_all=rewrite_all)