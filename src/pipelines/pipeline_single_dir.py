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


import KFoldHyperParameter

# The overarching pipeline to obtain all prerequisite data for the derrac pipeline
# Todo: Better standaradize the saving/loading
last_dct = []
def pipeline(file_name, space, bow, dct, classes, class_names, words_to_get, processed_folder, dims, kfold_hpam_dict, hpam_dict,
                     model_type="", dev_percent=0.2, rewrite_all=False, remove_stop_words=True,
                     score_metric="", auroc=False, dir_min_freq=0.001, dir_max_freq=0.95, name_of_class="", space_name="",
             classifier_fn="", mcm=None, top_scoring_dirs=2000, score_type="kappa", ppmi=None, dct_unchanged=None, pipeline_hpam_dict=None):
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

    hpam_save = SaveLoad(rewrite=True)

    # Folds and space are determined inside of the method for this hyper-parameter selection, as it is stacked
    hyper_param = KFoldHyperParameter.RecHParam(None, classes, class_names, pipeline_hpam_dict, kfold_hpam_dict, "dir", model_type,
                            file_name, None, processed_folder + "rank/", hpam_save, probability=False,
                            rewrite_model=rewrite_all, dev_percent=dev_percent,
                            data_type=data_type, score_metric=score_metric, auroc=auroc, matched_ids=matched_ids, end_fn_added=name_of_class,
                            mcm=mcm, hpam_params=[dct_unchanged, dct, bow, dir_min_freq, dir_max_freq, file_name, processed_folder,
                       words_to_get, space, name_of_class, model_type, classes, class_names, auroc, score_metric,
                       mcm, dev_percent, ppmi, kfold_hpam_dict])
    hyper_param.process_and_save()
    print("END OF SPACE")


def direction_pipeline(dct_unchanged, dct, bow, dir_min_freq, dir_max_freq, file_name, processed_folder,
                       words_to_get, space, name_of_class, model_type, classes, class_names, auroc, score_metric,
                       mcm, dev_percent, ppmi, kfold_hpam_dict, top_scoring_dir=None, top_scoring_freq=None):
    # Convert to hyper-parameter method, where hyper-parameters are:
    ##### dir_min_freq, dir_max freq
    ##### Scoring filters, aka top 200, top 400, etc, score-type
    dct_len_start = len(dct_unchanged.dfs.keys())
    if bow.shape[0] != len(dct.dfs.keys()) or len(dct.dfs.keys()) != ppmi.shape[0]:
        print("bow", bow.shape[0], "dct", len(dct.dfs.keys()), "ppmi", ppmi.shape[0])
        raise ValueError("Size of vocab and dict do not match")


    doc_amt = split.get_doc_amt(data_type)

    wl_save = SaveLoad(rewrite=rewrite_all)
    if dir_min_freq != -1 and dir_max_freq != -1:
        no_below = int(doc_amt * dir_min_freq)
        no_above = int(doc_amt * dir_max_freq)
        dir = LimitWords(file_name, wl_save, dct, bow, processed_folder +"directions/words/", words_to_get, no_below, no_above)
    else:
        # called this for laziness sake
        no_below = top_scoring_freq
        no_above = 0
        dir = LimitWordsNumeric(file_name, wl_save, dct, bow, processed_folder +"directions/words/", words_to_get, no_below)

    print("(For directions) Filtering all words that do not appear in", no_below, "documents")

    dir.process_and_save()
    words_to_get = dir.getBowWordDct()
    new_word_dict = dir.getNewWordDict()


    dir_save = SaveLoad(rewrite=rewrite_all)
    dir = GetDirections(bow, space, words_to_get, new_word_dict, dir_save, no_below, no_above, file_name , processed_folder + "directions/", LR=False)
    dir.process_and_save()
    binary_bow = np.asarray(dir.getNewBow().todense(), dtype=np.int32)
    binary_bow[binary_bow >= 1] = 1
    preds = dir.getPreds()
    words = dir.getWords()

    new_word2id_dict = {}
    for i in range(len(words)):
        new_word2id_dict[words[i]] = i

    score_save = SaveLoad(rewrite=rewrite_all)
    score = MultiClassScore(binary_bow, preds, None, file_name + "_" + str(no_below) + "_" + str(no_above) , processed_folder + "directions/score/", score_save, f1=True, auroc=False,
                    fscore=True, kappa=True, acc=True, class_names=words, verbose=False, directions=True, save_csv=True)
    score.process_and_save()
    s_dict = score.get()

    # Get rankings on directions save all of them in a word:ranking on entities format, and retrieve if already saved
    dirs = dir.getDirections()

    rank_save = SaveLoad(rewrite=rewrite_all)

    stream_rankings = False
    if (data_type == "sentiment" and no_below > 5000) or (no_below >= 10000 and data_type != "movies") or no_below > 10000:
        stream_rankings = True

    if stream_rankings: #and no_below > 5000) or no_below > 10000:
        rank = GetRankingsStreamed(dirs, space, new_word2id_dict,  rank_save,  file_name, processed_folder, no_below, no_above)
    else:
        rank = GetRankings(dirs, space, new_word2id_dict,  rank_save,  file_name, processed_folder, no_below, no_above)
    rank.process_and_save()
    rankings = rank.getRankings()

    dct_len_end =  len(dct_unchanged.dfs.keys())
    if dct_len_start != dct_len_end:
        raise ValueError("Dct has changed shape")

    # Get NDCG scores
    ndcg_save = SaveLoad(rewrite=rewrite_all)

    if stream_rankings: #and no_below > 5000) or no_below > 10000:
        ndcg = GetNDCGStreamed(rankings, ppmi, new_word2id_dict, dct_unchanged.token2id,  ndcg_save,  file_name, processed_folder + "rank/ndcg/", no_below, no_above)
    else:
        ndcg = GetNDCG(rankings, ppmi, new_word2id_dict, dct_unchanged.token2id,  ndcg_save,  file_name, processed_folder + "rank/ndcg/", no_below, no_above)
    ndcg.process_and_save()
    ndcg_scores = ndcg.getNDCG()

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

    score_array = [ndcg_scores, s_dict["f1"], s_dict["acc"], s_dict["kappa"]]
    sc_name_array = ["ndcg", "f1", "acc", "kappa"]
    # Filter directions based on the amount to filter in params and the score type to filter in params

    tsp = []
    tsrd = []
    rfn = []
    f1s = []
    for i in range(len(score_array)):
        dir_fn = file_name + "_" + sc_name_array[i] + "_" + str(top_scoring_dir) + "_" + str(no_below) + "_" + str(no_above)
        gtr_save = SaveLoad(rewrite=rewrite_all)
        if stream_rankings:
            gtr = GetTopScoringRanksStreamed(dir_fn, gtr_save, processed_folder + "rank/", score_array[i], top_scoring_dir, rankings, new_word2id_dict)
        else:
            gtr = GetTopScoringRanks(dir_fn, gtr_save, processed_folder + "rank/", score_array[i], top_scoring_dir, rankings, new_word2id_dict)
        gtr.process_and_save()
        fil_rank = gtr.getRank()

        split_ids = split.get_split_ids(data_type, matched_ids)
        x_train, y_train, x_test, y_test, x_dev, y_dev = split.split_data(fil_rank,
                                                                          classes, split_ids,
                                                                          dev_percent_of_train=dev_percent)
        if data_type == "placetypes" or data_type == "movies":
            dir_fn += "_" + name_of_class
        dir_fn += "_Fix"
        hpam_save = SaveLoad(rewrite=True)
        hyper_param = KFoldHyperParameter.HParam(class_names, kfold_hpam_dict, model_type, dir_fn, processed_folder + "rank/", hpam_save,
                             False, rewrite_model=True, x_train=x_train, y_train=y_train, x_test=x_test,
                             y_test=y_test, x_dev=x_dev, y_dev=y_dev, score_metric=score_metric, auroc=auroc, mcm=mcm, dim_names=words)
        hyper_param.process_and_save()

        tsp.append(hyper_param.getTopScoringParams())
        tsrd.append(hyper_param.getTopScoringRowData())
        rfn.append(gtr.rank.file_name)
        f1s.append(hyper_param.getTopScoringRowData()[1][1]) # F1 Score for the current scoring metric



    best_ind = np.flipud(np.argsort(f1s))[0]
    top_params = tsp[best_ind]
    top_row_data = tsrd[best_ind]
    top_rank = rfn[best_ind]

    all_r = np.asarray(tsrd).transpose()
    rows = all_r[1]
    cols = np.asarray(rows.tolist()).transpose()
    col_names = all_r[0][0]
    key = all_r[2]
    dt.write_csv(processed_folder + "rank/score/csv_averages/" +file_name+ "_" + str(no_above) + "_" + str(no_below)
                 + "reps"+model_type+"_" + name_of_class + ".csv", col_names, cols, key)
    print("Pipeline completed, saved as, "+ processed_folder + "rank/score/csv_averages/" +file_name+ "_" + str(no_above) + "_" + str(no_below)
                 + "reps"+model_type+"_" + name_of_class + ".csv")
    return top_params, top_row_data, top_rank





def main(data_type, raw_folder, processed_folder,proj_folder="",  grams=0, model_type="LinearSVM", dir_min_freq=0.001,
         dir_max_freq=0.95, dev_percent=0.2, score_metric="avg_f1", max_depth=None, multiclass="OVR", LR=False, bonus_fn="",
         rewrite_all = False, hp_top_freq=None, hp_top_dir=None):
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

    dims = [100, 50, 200]
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

    pipeline_hpam_dict = {"top_freq": hp_top_freq, "top_dir": hp_top_dir}

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
        classes_process = util.classify.ProcessClasses(None, None, pipeline_fn, processed_folder, bowmin, no_below,
                                                       no_above, classes_freq_cutoff, True, classes_save, name_of_class[ci])

        classes = classes_process.getClasses()
        class_names = classes_process.getClassNames()

        corp_save = SaveLoad(rewrite=False)
        p_corpus = process_corpus.Corpus(None, classes, name_of_class[ci], pipeline_fn, processed_folder,
                                         bowmin,
                                         no_below, no_above, True, corp_save)
        bow = p_corpus.getBow()
        word_list = p_corpus.getAllWords()
        classes = p_corpus.getClasses()

        ppmi_save = SaveLoad(rewrite=False)
        ppmi_identifier = "_ppmi"
        ppmi_fn = pipeline_fn + ppmi_identifier

        ppmi_unfiltered = ppmi.PPMI(p_corpus.getBow(), None, processed_folder + "bow/" + ppmi_fn, ppmi_save)
        ppmi_unfiltered.process_and_save()
        ppmi_unf_matrix = ppmi_unfiltered.getMatrix().transpose()

        for i in range(len(dims)):
            spaces = []
            space_names = []

            if data_type != "sentiment":
                mds_identifier = "_" + str(dims[i]) + "_MDS"
                mds_fn = pipeline_fn + mds_identifier
                import_fn = processed_folder + "rep/mds/" + mds_fn + ".npy"
                mds_space = dt.import2dArray(import_fn)
                spaces.append(mds_space)

                metadata_fn = processed_folder + "bow/metadata/" + "num_stw_remove.npy"
                """
                if len(mds_space) != 18302:
                    del_ids = np.load(metadata_fn)
                    mds_space = np.delete(mds_space, del_ids, axis=0)
                    np.save(import_fn,mds_space)
                """
                space_names.append(mds_fn)

            awv_identifier = "_" + str(dims[i]) + "_AWVEmp"
            awv_fn = pipeline_fn + awv_identifier
            awv_instance = awv.AWV(None, dims[i], awv_fn, processed_folder + "rep/awv/", SaveLoad(rewrite=False))
            awv_instance.process_and_save()
            awv_space = awv_instance.getRep()
            spaces.append(awv_space)
            space_names.append(awv_fn)

            pca_identifier = "_" + str(dims[i]) + "_PCA"
            pca_fn = pipeline_fn + pca_identifier
            pca_instance = pca.PCA(None, None, dims[i],
                                   pca_fn, processed_folder + "rep/pca/", SaveLoad(rewrite=False))
            pca_instance.process_and_save()
            spaces.append(pca_instance.getRep())
            space_names.append(pca_fn)


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
                hyper_param = KFoldHyperParameter.RecHParam(None, classes, None, hpam_dict, hpam_dict, "d2v", "LinearSVM",
                                        doc2vec_fn, classifier_fn, processed_folder + "rep/", SaveLoad(rewrite=False), data_type=data_type,
                                        score_metric=score_metric)
                d2v_space, __unused = hyper_param.getTopScoringSpace()
                spaces.append(d2v_space)
                space_names.append(doc2vec_fn)


            for s in range(len(spaces)):
                if s > 0:
                    if len(spaces[s]) != len(spaces[s-1]):
                        raise ValueError("Space len is incorrect", s, len(spaces[s]), space_names[s])
                if len(classes) != len(spaces[s]):
                    raise ValueError("Length of classes does not equal length of spaces")
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
                    pipeline(final_fn, spaces[s], bow, dct, classes, class_names, word_list, processed_folder, dims, kfold_hpam_dict, hpam_dict,
                 model_type=model_type, dev_percent=dev_percent, rewrite_all=rewrite_all, remove_stop_words=True,
                 score_metric=score_metric, auroc=False, dir_min_freq=dir_min_freq, dir_max_freq=dir_max_freq, name_of_class=name_of_class[ci], classifier_fn = classifier_fn,
                             mcm=multi_class_method, ppmi=ppmi_unf_matrix, dct_unchanged=dct_unchanged, pipeline_hpam_dict=pipeline_hpam_dict, space_name=space_names[s])
                else:
                    classifier_fn = pipeline_fn + "_" + multiclass
                    pipeline(final_fn, spaces[s], bow, dct, classes, class_names, word_list, processed_folder, dims, kfold_hpam_dict, hpam_dict,
                     model_type=model_type, dev_percent=dev_percent, rewrite_all=rewrite_all, remove_stop_words=True,
                     score_metric=score_metric, auroc=False, dir_min_freq=dir_min_freq, dir_max_freq=dir_max_freq, name_of_class=name_of_class[ci], classifier_fn = classifier_fn,
                             mcm=multi_class_method, ppmi=ppmi_unf_matrix, dct_unchanged=dct_unchanged, pipeline_hpam_dict=pipeline_hpam_dict, space_name=space_names[s])

max_depths = [3]
classifiers = [ "DecisionTree3" ]
data_type = "reuters"
doLR = False
dminf = -1
dmanf = -1

if data_type == "placetypes":
    hp_top_freq = [5000, 10000, 20000]
    hp_top_dir = [1000, 2000]
elif data_type == "reuters":
    hp_top_freq = [5000, 10000, 20000]
    hp_top_dir = [1000,  2000]
elif data_type == "sentiment":
    hp_top_freq = [5000,  10000, 20000]
    hp_top_dir = [1000, 2000]
elif data_type == "newsgroups":
    hp_top_freq = [5000,  10000, 20000]
    hp_top_dir = [1000,  2000]
elif data_type == "movies":
    hp_top_freq = [5000, 10000, 20000]
    hp_top_dir = [1000,  2000]

multi_class_method = "OVR"
bonus_fn = ""
rewrite_all=False
if __name__ == '__main__':
    for i in range(len(classifiers)):
        main(data_type, "../../data/raw/"+data_type+"/",  "../../data/processed/"+data_type+"/", proj_folder="../../data/proj/"+data_type+"/",
                                grams=0, model_type=classifiers[i], dir_min_freq=dminf, dir_max_freq=dmanf, dev_percent=0.2,
                                score_metric="avg_f1", max_depth=max_depths[i], multiclass=multi_class_method, LR=doLR, bonus_fn=bonus_fn, rewrite_all=rewrite_all,
             hp_top_dir=hp_top_dir, hp_top_freq=hp_top_freq)