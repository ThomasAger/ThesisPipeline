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
from project.get_directions import GetDirections
from score.classify import MultiClassScore
from project.get_rankings import GetRankings, GetRankingsStreamed
from project.get_ndcg import GetNDCG, GetNDCGStreamed
from rep import pca, ppmi, awv
from project.get_tsr import GetTopScoringRanks, GetTopScoringRanksStreamed
from project.get_tsd import GetTopScoringDirs, GetTopScoringDirsStreamed
from util import py
def get_dp(space, direction):
    ranking = np.empty(len(space))
    for i in range(len(space)):
        ranking[i] = np.dot(direction, space[i])
    return ranking
import KFoldHyperParameter

# The overarching pipeline to obtain all prerequisite data for the derrac pipeline
# Todo: Better standaradize the saving/loading
def pipeline(corpus, classes, class_names, file_name, output_folder, dims, kfold_hpam_dict, hpam_dict, bowmin,
             no_below_fraction, no_above, classes_freq_cutoff, model_type, dev_percent, data_type, rewrite_all=False,
             remove_stop_words=False, auroc=False, score_metric="avg_f1", corpus_fn="", name_of_class="",
             mcm=MultiOutputClassifier, classifier_fn="", processed_folder=""):
    probability = False
    if auroc is True:
        probability = True

    doc_amt = split.get_doc_amt(data_type)
    no_below = int(doc_amt * no_below_fraction)
    print("Filtering all words that do not appear in", no_below, "documents")
    classes_save = SaveLoad(rewrite=rewrite_all)
    classes_process = util.classify.ProcessClasses(classes, class_names, file_name, output_folder, bowmin, no_below,
                                                   no_above, classes_freq_cutoff, remove_stop_words, classes_save,
                                                   name_of_class)
    classes_process.process_and_save()
    classes = classes_process.getClasses()
    class_names = classes_process.getClassNames()

    # Process and save corpus
    corpus_save = SaveLoad(rewrite=rewrite_all)
    if data_type == "placetypes" or data_type == "movies":
        print("Processing corpus")
        p_corpus = process_corpus.StreamedCorpus(classes, name_of_class, file_name, output_folder, bowmin, no_below,
                                                 no_above, remove_stop_words, corpus_save,
                                                 corpus_fn_to_stream=corpus_fn)
    else:
        print("Processing corpus")
        p_corpus = process_corpus.Corpus(corpus, classes, name_of_class, file_name, output_folder, bowmin, no_below,
                                         no_above, remove_stop_words, corpus_save)

    p_corpus.process_and_save()
    p_classes = p_corpus.getClasses()
    dct = p_corpus.getBowDct()
    dct_unchanged = p_corpus.getBowDct()
    bow = p_corpus.getBow()
    matched_ids = []
    try:
        class_entities = dt.import1dArray(output_folder + "classes/" + name_of_class + "_entities.txt")
        entity_names = dt.import1dArray(output_folder + "corpus/entity_names.txt")
        for i in range(len(class_entities)):
            for j in range(len(entity_names)):
                if class_entities[i] == entity_names[j]:
                    matched_ids.append(j)
                    break
    except FileNotFoundError:
        matched_ids = None

    # Get the PPMI values
    ppmi_save = SaveLoad(rewrite=rewrite_all)
    ppmi_identifier = "_ppmi"
    ppmi_fn = file_name + ppmi_identifier
    classify_ppmi_fn = classifier_fn + ppmi_identifier

    hyper_param = HParam(class_names, kfold_hpam_dict, model_type, classify_ppmi_fn,
                         output_folder + "rep/", classes_save, probability, rewrite_model=rewrite_all,
                         score_metric=score_metric, auroc=auroc,
                         mcm=mcm)
    if os.path.exists(hyper_param.averaged_csv_data.file_name) is False:

        ppmi_unfiltered = ppmi.PPMI(p_corpus.getBow(), doc_amt, output_folder + "bow/" + ppmi_fn, ppmi_save)
        ppmi_unfiltered.process_and_save()
        ppmi_unf_matrix = ppmi_unfiltered.getMatrix()

        ppmi_save = SaveLoad(rewrite=rewrite_all)
        ppmi_filtered = ppmi.PPMI(p_corpus.getFilteredBow(), doc_amt,
                                  output_folder + "bow/NB_" + str(no_below) + "_NA_" + str(no_above) + ppmi_fn, ppmi_save)
        ppmi_filtered.process_and_save()
        ppmi_filtered_matrix = ppmi_filtered.getMatrix()

        # Get the dev splits
        split_ids = split.get_split_ids(data_type, matched_ids)
        x_train, y_train, x_test, y_test, x_dev, y_dev = split.split_data(ppmi_filtered_matrix.toarray(), p_classes,
                                                                          split_ids, dev_percent_of_train=dev_percent)

        all_test_result_rows = []

        hpam_save = SaveLoad(rewrite=rewrite_all)
        hyper_param = HParam(class_names, kfold_hpam_dict, model_type, classify_ppmi_fn,
                             output_folder + "rep/", hpam_save, probability, rewrite_model=rewrite_all, x_train=x_train,
                             y_train=y_train, x_test=x_test, y_test=y_test, x_dev=x_dev, y_dev=y_dev,
                             score_metric=score_metric, auroc=auroc,
                             mcm=mcm)
        hyper_param.process_and_save()

        all_test_result_rows.append(hyper_param.getTopScoringRowData())


    pca_save = SaveLoad(rewrite=rewrite_all)
    pca_identifier = "_" + str(200) + "_PCA"
    pca_fn = file_name + pca_identifier
    classify_pca_fn = classifier_fn + pca_identifier

    hpam_save = SaveLoad(rewrite=rewrite_all)
    hyper_param = HParam(hpam_dict=kfold_hpam_dict, model_type=model_type, file_name=classify_pca_fn,
                         output_folder=output_folder + "rep/", save_class=hpam_save, rewrite_model=rewrite_all,
                         score_metric=score_metric, mcm=mcm)
    if not hyper_param.save_class.exists(hyper_param.popo_array) or hyper_param.save_class.rewrite is True:
        pca_instance = pca.PCA(ppmi_unf_matrix, doc_amt, 200,
                               pca_fn, output_folder + "rep/pca/", pca_save)
        pca_instance.process_and_save()
        space = pca_instance.getRep()

        split_ids = split.get_split_ids(data_type, matched_ids)
        x_train, y_train, x_test, y_test, x_dev, y_dev = split.split_data(space,
                                                                          p_classes, split_ids,
                                                                          dev_percent_of_train=dev_percent)
    else:
        pca_instance = pca.PCA(ppmi_unf_matrix, doc_amt, 200,
                               pca_fn, output_folder + "rep/pca/", pca_save)
        space = pca_instance.getRep()

    hpam_save = SaveLoad(rewrite=rewrite_all)
    hyper_param = HParam(class_names,
                         kfold_hpam_dict, model_type, classify_pca_fn,
                         output_folder + "rep/", hpam_save, probability, rewrite_model=rewrite_all,
                         x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, x_dev=x_dev, y_dev=y_dev,
                         score_metric=score_metric, auroc=auroc,
                         mcm=mcm)
    hyper_param.process_and_save()

    all_test_result_rows.append(hyper_param.getTopScoringRowData())

    # Make the combined csv of all space types
    # Make the combined CSV of all the dims of all the space types
    all_r = np.asarray(all_test_result_rows).transpose()
    rows = all_r[1]
    for i in range(len(rows)):
        if len(rows[i]) == 7:
            rows[i] = rows[i][:5]

            if len(rows[i]) != 5:
                print(len(rows[i]))
                raise ValueError("No, bro.")
    cols = np.asarray(rows.tolist()).transpose()
    col_names = all_r[0][0]
    key = all_r[2]
    dt.write_csv(
        output_folder + "rep/score/csv_final/" + file_name + "reps" + model_type + "_" + name_of_class + ".csv",
        col_names, cols, key)
    print("a")

    ##### dir_min_freq, dir_max freq
    ##### Scoring filters, aka top 200, top 400, etc, score-type


    wl_save = SaveLoad(rewrite=True)

    # called this for laziness sake
    no_below = 2000
    top_scoring_dir = 2000
    no_above = 0

    dir = LimitWordsNumeric(file_name, wl_save, dct, bow, processed_folder + "directions/words/", None, no_below)

    print("(For directions) Filtering all words that do not appear in", no_below, "documents")

    dir.process_and_save()
    words_to_get = dir.getBowWordDct()
    new_word_dict = dir.getNewWordDict()

    dir_save = SaveLoad(rewrite=rewrite_all)
    dir = GetDirections(bow, space, words_to_get, new_word_dict, dir_save, no_below, no_above, file_name,
                        processed_folder + "directions/", LR=False)
    dir.process_and_save()
    # Get rankings on directions save all of them in a word:ranking on entities format, and retrieve if already saved
    dirs = dir.getDirections()

    binary_bow = np.asarray(dir.getNewBow().todense(), dtype=np.int32)
    binary_bow[binary_bow >= 1] = 1
    preds = dir.getPreds()
    words = dir.getWords()

    new_word2id_dict = {}
    for i in range(len(words)):
        new_word2id_dict[words[i]] = i

    score_save = SaveLoad(rewrite=rewrite_all)
    score = MultiClassScore(binary_bow, preds, None, file_name + "_" + str(no_below) + "_" + str(no_above),
                            processed_folder + "directions/score/", score_save, f1=True, auroc=False,
                            fscore=True, kappa=True, acc=True, class_names=words, verbose=False, directions=True,
                            save_csv=True)
    score.process_and_save()
    s_dict = score.get()

    rank_save = SaveLoad(rewrite=rewrite_all)

    stream_rankings = True

    rank = GetRankings(dirs, space, new_word2id_dict, rank_save, file_name, processed_folder, no_below, no_above)
    rank.process_and_save()
    rankings = rank.getRankings()
    if len(rankings) < 100:
        if False in np.isclose(dt.importFirstLineOfTextFileAsFloat(rankings), get_dp(space, dirs[0])):
            raise ValueError("Rankings do not match")
    elif stream_rankings is False:
        print("kay")
        if False in np.isclose(rankings[0], get_dp(space, dirs[0])):
            raise ValueError("Rankings do not match")
    dct_len_start = len(dct_unchanged.dfs.keys())
    dct_len_end = len(dct_unchanged.dfs.keys())
    if dct_len_start != dct_len_end:
        raise ValueError("Dct has changed shape")

    score_save = SaveLoad(rewrite=rewrite_all)
    score = MultiClassScore(binary_bow, preds, None, file_name + "_" + str(no_below) + "_" + str(no_above),
                            processed_folder + "directions/score/", score_save, f1=True, auroc=False,
                            fscore=True, kappa=True, acc=True, class_names=words, verbose=False, directions=True,
                            save_csv=True)
    score.process_and_save()
    s_dict = score.get()

    """
    # Get NDCG scores
    ndcg_save = SaveLoad(rewrite=True)

    if stream_rankings:  # and no_below > 5000) or no_below > 10000:
        ndcg = GetNDCGStreamed(rankings, ppmi_unf_matrix, new_word2id_dict, dct_unchanged.token2id, ndcg_save, file_name,
                               processed_folder + "rank/ndcg/", no_below, no_above)
    else:
        ndcg = GetNDCG(rankings, ppmi_unf_matrix, new_word2id_dict, dct_unchanged.token2id, ndcg_save, file_name,
                       processed_folder + "rank/ndcg/", no_below, no_above)
    ndcg.process_and_save()
    ndcg_scores = ndcg.getNDCG()
    """
    matched_ids = []
    try:
        class_entities = dt.import1dArray(processed_folder + "classes/" + name_of_class + "_entities.txt")
        entity_names = dt.import1dArray(processed_folder + "corpus/entity_names.txt")
        for i in range(len(class_entities)):
            for j in range(len(entity_names)):
                if class_entities[i] == entity_names[j]:
                    matched_ids.append(j)
                    break
    except FileNotFoundError:
        matched_ids = None

    # Filter directions based on the amount to filter in params and the score type to filter in params

    tsp = []
    tsrd = []
    rfn = []
    dfn = []
    f1s = []
    fn_final = []

    split_ids = split.get_split_ids(data_type, matched_ids)
    x_train, y_train, x_test, y_test, x_dev, y_dev = split.split_data(rankings.transpose(),
                                                                      classes, split_ids,
                                                                      dev_percent_of_train=dev_percent,
                                                                      data_type=data_type)

    hpam_save = SaveLoad(rewrite=rewrite_all)
    hyper_param = KFoldHyperParameter.HParam(class_names, kfold_hpam_dict, model_type, file_name,
                                             processed_folder + "rank/", hpam_save,
                                             False, rewrite_model=rewrite_all, x_train=x_train, y_train=y_train,
                                             x_test=x_test, feature_names=words,
                                             y_test=y_test, x_dev=x_dev, y_dev=y_dev, score_metric=score_metric,
                                             auroc=auroc, mcm=mcm, dim_names=words)
    hyper_param.process_and_save()

    tsp.append(hyper_param.getTopScoringParams())
    tsrd.append(hyper_param.getTopScoringRowData())
    f1s.append(hyper_param.getTopScoringRowData()[1][1])  # F1 Score for the current scoring metric

    all_r = np.asarray(tsrd).transpose()
    rows = all_r[1]
    cols = np.asarray(rows.tolist()).transpose()
    col_names = all_r[0][0]
    key = all_r[2]
    dt.write_csv(processed_folder + "rank/score/csv_averages/" + file_name + "_" + str(no_above) + "_" + str(no_below)
                 + "reps" + model_type + "_" + name_of_class + ".csv", col_names, cols, key)
    print("Pipeline completed, saved as, " + processed_folder + "rank/score/csv_averages/" + file_name + "_" + str(
        no_above) + "_" + str(no_below)
          + "reps" + model_type + "_" + name_of_class + ".csv")


def main(data_type, raw_folder, processed_folder, proj_folder="", grams=0, model_type="LinearSVMNOvR", no_below=0.001,
         no_above=0.95, classes_freq_cutoff=100, bowmin=2, dev_percent=0.2, score_metric="avg_f1", max_depth=None,
         multiclass="MOP"):
    corpus_fn = ""
    if data_type == "newsgroups":
        corpus_fn = processed_folder + "corpus/" + "num_stw_corpus_processed.txt"
        newsgroups = fetch_20newsgroups(subset='all', shuffle=False, remove=("headers", "footers", "quotes"))
        corpus = newsgroups.data
        classes = newsgroups.target
        class_names = newsgroups.target_names
        name_of_class = "Newsgroups"
        print("newsgroups!")
    elif data_type == "sentiment":
        corpus_fn = processed_folder + "corpus/" + "num_stw_corpus_processed.txt"
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=0, skip_top=0, index_from=0, seed=113)
        corpus = np.asarray(np.concatenate((x_train, x_test), axis=0))
        classes = np.asarray(np.concatenate((y_train, y_test), axis=0))
        corpus = np.asarray(util.text_utils.makeCorpusFromIds(corpus, imdb.get_word_index()))
        class_names = ["sentiment"]
        classes_freq_cutoff = 0
        name_of_class = "Sentiment"
    elif data_type == "movies":
        corpus_fn = raw_folder + "corpus.txt"
        corpus = None
        genres = np.load(raw_folder + "/genres/class-all.npy")
        genre_names = dt.import1dArray(raw_folder + "/genres/names.txt")
        keywords = np.load(raw_folder + "/keywords/class-all.npy")
        keyword_names = dt.import1dArray(raw_folder + "/keywords/names.txt")
        ratings = np.load(raw_folder + "/ratings/class-all.npy")
        rating_names = dt.import1dArray(raw_folder + "/ratings/names.txt")
        classes = [genres, keywords, ratings]
        class_names = [genre_names, keyword_names, rating_names]
        name_of_class = ["Genres", "Keywords", "Ratings"]
    elif data_type == "placetypes":
        corpus_fn = raw_folder + "corpus.txt"
        corpus = None
        foursquare = dt.import2dArray(raw_folder + "/Foursquare/class-all", "i")
        foursquare_names = dt.import1dArray(raw_folder + "/Foursquare/names.txt", "s")
        geonames = dt.import2dArray(raw_folder + "/Geonames/class-all", "i")
        geonames_names = dt.import1dArray(raw_folder + "/Geonames/names.txt", "s")
        opencyc = dt.import2dArray(raw_folder + "/OpenCYC/class-all", "i")
        opencyc_names = dt.import1dArray(raw_folder + "/OpenCYC/names.txt", "s")
        classes = [foursquare, geonames, opencyc]
        class_names = [foursquare_names, geonames_names, opencyc_names]
        name_of_class = ["Foursquare", "Geonames", "OpenCYC"]
        classes_freq_cutoff = 0

    elif data_type == "reuters":
        corpus_fn = processed_folder + "corpus/" + "num_stw_corpus_processed.txt"
        classes_freq_cutoff = 100
        corpus = dt.import1dArray(raw_folder + "duplicate_removed_docs.txt")
        classes = dt.import2dArray(raw_folder + "unique_classes.txt", "i")
        class_names = dt.import1dArray(raw_folder + "class_names.txt")
        name_of_class = "Reuters"

    elif data_type == "anime":
        corpus_fn = processed_folder + "corpus/" + "num_stw_corpus_processed.txt"
        classes_freq_cutoff = 10
        corpus = np.load(raw_folder + "corpus.npy")
        classes = np.load(raw_folder + "classes.npy")
        class_names = list(range(len(classes[0])))
        for i in range(len(class_names)):
            class_names[i] = str(class_names[i])
        name_of_class = "Genres"

    elif data_type == "mafiascum":
        corpus_fn = processed_folder + "corpus/" + "num_stw_corpus_processed.txt"
        corpus = np.load(raw_folder + "user_docs.npy")
        classes = dt.import1dArray(raw_folder + "classes.txt", "i")
        class_names = dt.import1dArray(raw_folder + "usernames.txt")
        name_of_class = "scum"
        classes_freq_cutoff = 0

    window_size = [5, 10, 15]
    min_count = [1, 5, 10]
    train_epoch = [50, 100, 200]

    dims = [50, 100, 200]
    balance_params = ["balanced", None]
    C_params = [1.0, 0.01, 0.001, 0.0001]
    gamma_params = [1.0, 0.01, 0.001, 0.0001]

    n_estimators = [1000, 2000]
    max_features = [None, 'auto', 'log2']
    criterion = ["gini", "entropy"]
    max_depth = [max_depth]
    bootstrap = [True, False]
    min_samples_leaf = [1]
    min_samples_split = [2]

    # Run a pipeline that retains numbers and removes stopwords

    if model_type == "GaussianSVM" or model_type == "GaussianSVMMultiClass":
        kfold_hpam_dict = {"C": C_params,
                           "class_weight": balance_params,
                           "gamma": gamma_params}
    elif model_type == "LinearSVM":
        kfold_hpam_dict = {"C": C_params,
                           "class_weight": balance_params}
    elif model_type == "RandomForest":
        kfold_hpam_dict = {"n_estimators": n_estimators,
                           "max_features": max_features,
                           "class_weight": balance_params,
                           "criterion": criterion,
                           "max_depth": max_depth,
                           "bootstrap": bootstrap,
                           "min_samples_leaf": min_samples_leaf,
                           "min_samples_split": min_samples_split}
    elif model_type[:12] == "DecisionTree":
        kfold_hpam_dict = {"max_features": max_features,
                           "class_weight": balance_params,
                           "criterion": criterion,
                           "max_depth": max_depth}

    multi_class_method = None
    if multiclass == "MOP":
        multi_class_method = MultiOutputClassifier
    elif multiclass == "OVR":
        multi_class_method = OneVsRestClassifier
    elif multiclass == "OVO":
        multi_class_method = OneVsOneClassifier
    elif multiclass == "OCC":
        multi_class_method = OutputCodeClassifier

    hpam_dict = {"window_size": window_size,
                 "min_count": min_count,
                 "train_epoch": train_epoch}
    pipeline_fn = "num_stw"
    if data_type == "movies" or data_type == "placetypes":
        for i in range(len(classes)):
            classifier_fn = pipeline_fn + "_" + name_of_class[i] + "_" + multiclass
            pipeline(corpus, classes[i], class_names[i], pipeline_fn, processed_folder, dims, kfold_hpam_dict,
                     hpam_dict, bowmin,
                     no_below,
                     no_above, classes_freq_cutoff, model_type, dev_percent, data_type, rewrite_all=False,
                     remove_stop_words=True,
                     score_metric=score_metric, auroc=False, corpus_fn=corpus_fn, name_of_class=name_of_class[i],
                     classifier_fn=classifier_fn, mcm=multi_class_method, processed_folder=processed_folder)
    else:
        classifier_fn = pipeline_fn + "_" + multiclass
        pipeline(corpus, classes, class_names, pipeline_fn, processed_folder, dims, kfold_hpam_dict, hpam_dict, bowmin,
                 no_below,
                 no_above, classes_freq_cutoff, model_type, dev_percent, data_type, rewrite_all=False,
                 remove_stop_words=True, score_metric=score_metric, auroc=False,
                 corpus_fn=corpus_fn, name_of_class=name_of_class, classifier_fn=classifier_fn, mcm=multi_class_method, processed_folder=processed_folder)


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

    max_depths = [None, None, 3, 2, 1]
    classifiers = ["DecisionTree3", "LinearSVM"]
    data_type = ["mafiascum"]
    if __name__ == '__main__':
        for j in range(len(data_type)):
            for i in range(len(classifiers)):
                print(data_type[j])
                main(data_type[j], "../../data/raw/" + data_type[j] + "/", "../../data/processed/" + data_type[j] + "/",
                     proj_folder="../../data/proj/" + data_type[j] + "/",
                     grams=0, model_type=classifiers[i], no_below=0.001, no_above=0.95, classes_freq_cutoff=100,
                     bowmin=2, dev_percent=0.2,
                     score_metric="avg_f1", max_depth=max_depths[i], multiclass="OVR")