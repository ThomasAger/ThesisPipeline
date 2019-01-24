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

# The overarching pipeline to obtain all prerequisite data for the derrac pipeline
# Todo: Better standaradize the saving/loading
def pipeline(corpus, classes, class_names, file_name, output_folder, dims, kfold_hpam_dict, hpam_dict, bowmin,
             no_below_fraction, no_above, classes_freq_cutoff, model_type, dev_percent, rewrite_all=False,
             remove_stop_words=False,  auroc=False, score_metric="avg_f1", corpus_fn="", name_of_class=""):
    classifier_fn = file_name + "_" + name_of_class + "_"

    probability = False
    if auroc is True:
        probability = True

    doc_amt = split.get_doc_amt(data_type)
    no_below = int(doc_amt * no_below_fraction)
    print("Filtering all words that do not appear in", no_below, "documents")
    classes_save = SaveLoad(rewrite=rewrite_all)
    classes_process = process_corpus.ProcessClasses(classes, class_names, file_name, output_folder, bowmin, no_below,
                                         no_above, classes_freq_cutoff, remove_stop_words, classes_save, name_of_class)
    classes_process.process_and_save()
    classes = classes_process.filtered_classes.value
    class_names = classes_process.filtered_class_names.value



    # Process and save corpus
    corpus_save = SaveLoad(rewrite=rewrite_all)
    if data_type == "placetypes" or data_type == "movies":
        p_corpus = process_corpus.StreamedCorpus(classes, name_of_class,  file_name, output_folder, bowmin, no_below,
                                         no_above, remove_stop_words, corpus_save, corpus_fn_to_stream=corpus_fn)
    else:
        p_corpus = process_corpus.Corpus(corpus,  classes,name_of_class, file_name, output_folder, bowmin, no_below,
                                         no_above, remove_stop_words, corpus_save)
    p_corpus.process_and_save()

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
    ppmi_fn = file_name + "_ppmi"
    ppmi_unfiltered = ppmi.PPMI(p_corpus.bow.value, doc_amt, output_folder + "bow/" + ppmi_fn, ppmi_save)
    ppmi_unfiltered.process_and_save()
    ppmi_unf_matrix = ppmi_unfiltered.ppmi_matrix.value


    ppmi_save = SaveLoad(rewrite=rewrite_all)
    ppmi_filtered = ppmi.PPMI(p_corpus.filtered_bow.value, doc_amt, output_folder + "bow/NB_" + str(no_below) + "_NA_" + str(no_above) + ppmi_fn, ppmi_save)
    ppmi_filtered.process_and_save()
    ppmi_filtered_matrix = ppmi_filtered.ppmi_matrix.value

    # Get the dev splits
    split_ids = split.get_split_ids(data_type, matched_ids)
    x_train, y_train, x_test, y_test, x_dev, y_dev = split.split_data(ppmi_filtered_matrix.toarray(), p_corpus.classes.value, split_ids, dev_percent_of_train=dev_percent)

    all_test_result_rows = []

    hpam_save = SaveLoad(rewrite=rewrite_all)
    hyper_param = HParam(class_names,
                                      kfold_hpam_dict, model_type, ppmi_fn,
                                      output_folder, hpam_save, probability, rewrite_model=rewrite_all, x_train=x_train,
                         y_train=y_train, x_test=x_test, y_test=y_test, x_dev=x_dev, y_dev=y_dev, score_metric=score_metric, auroc=auroc)
    hyper_param.process_and_save()

    all_test_result_rows.append(hyper_param.top_scoring_row_data.value)

    # Creating and testing spaces, MDS not included in the creation process
    for i in range(len(dims)):
        pca_save = SaveLoad(rewrite=rewrite_all)
        pca_identifier = "_" + str(dims[i]) + "_PCA"
        pca_fn = file_name + pca_identifier
        classify_pca_fn = classifier_fn + pca_identifier
        pca_instance = pca.PCA(ppmi_unf_matrix, doc_amt, dims[i],
                              pca_fn,  output_folder + "rep/pca/", pca_save)
        pca_instance.process_and_save()
        pca_space = pca_instance.rep.value

        split_ids = split.get_split_ids(data_type, matched_ids)
        x_train, y_train, x_test, y_test, x_dev, y_dev = split.split_data(pca_space,
                                                                          p_corpus.classes.value, split_ids,
                                                                          dev_percent_of_train=dev_percent)
        hpam_save = SaveLoad(rewrite=True)
        hyper_param = HParam(class_names,
                                          kfold_hpam_dict, model_type, classify_pca_fn,
                                     output_folder, hpam_save, probability, rewrite_model=rewrite_all,
                             x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, x_dev=x_dev, y_dev=y_dev, score_metric=score_metric, auroc=auroc)
        hyper_param.process_and_save()

        all_test_result_rows.append(hyper_param.top_scoring_row_data.value)

        wv_path = "D:/PhD/Code/ThesisPipeline/ThesisPipeline/data/raw/glove/glove.6B." + str(dims[i]) + 'd.txt'
        wv_path_d2v = "D:/PhD/Code/ThesisPipeline/ThesisPipeline/data/raw/glove/glove.6B.300d.txt"


        # We only have word-vectors of size 50, 100, 200 and 300
        if dims[i] != 20 and data_type != "movies":
            awv_save = SaveLoad(rewrite=rewrite_all)
            awv_identifier =  "_" + str(dims[i]) + "_AWVEmp"
            awv_fn = file_name + awv_identifier
            classify_awv_fn = classifier_fn + awv_identifier
            awv_instance = awv.AWV(p_corpus.split_corpus.value, dims[i], awv_fn, output_folder + "rep/awv/" , awv_save, wv_path=wv_path)
            awv_instance.process_and_save()
            awv_space = awv_instance.rep.value

            split_ids = split.get_split_ids(data_type, matched_ids)
            x_train, y_train, x_test, y_test, x_dev, y_dev = split.split_data(awv_space,
                                                                              p_corpus.classes.value, split_ids,
                                                                              dev_percent_of_train=dev_percent)

            hpam_save = SaveLoad(rewrite=rewrite_all)
            hyper_param = HParam(class_names,
                                              kfold_hpam_dict, model_type, classify_awv_fn,
                                         output_folder, hpam_save, probability, rewrite_model=rewrite_all, x_train=x_train,
                                 y_train=y_train, x_test=x_test, y_test=y_test, x_dev=x_dev, y_dev=y_dev, score_metric=score_metric, auroc=auroc)
            hyper_param.process_and_save()

            all_test_result_rows.append(hyper_param.top_scoring_row_data.value)

            if data_type != "sentiment" :
                mds_identifier =  "_" + str(dims[i])+"_MDS"
                mds_fn = file_name + mds_identifier
                classify_mds_fn = classifier_fn + mds_identifier
                import_fn = output_folder + "rep/mds/"+mds_fn+".npy"
                mds = dt.import2dArray(import_fn)

                split_ids = split.get_split_ids(data_type, matched_ids)
                x_train, y_train, x_test, y_test, x_dev, y_dev = split.split_data(mds,
                                                                                  p_corpus.classes.value, split_ids,
                                                                                  dev_percent_of_train=dev_percent)

                hpam_save = SaveLoad(rewrite=rewrite_all)
                hyper_param = HParam(class_names,
                                                  kfold_hpam_dict, model_type, classify_mds_fn,
                                             output_folder, hpam_save, probability, rewrite_model=rewrite_all, x_train=x_train,
                                     y_train=y_train, x_test=x_test, y_test=y_test, x_dev=x_dev, y_dev=y_dev, score_metric=score_metric, auroc=auroc)
                hyper_param.process_and_save()
                all_test_result_rows.append(hyper_param.top_scoring_row_data.value)
            """
            awvW_save = SaveLoad(rewrite=rewrite_all)
            awvW_fn = file_name + "_" + str(dims[i]) + "_awvW"

            awvW_instance = awv.AWVw(ppmi_unf_matrix, p_corpus.id2token.value, dims[i], awvW_fn, output_folder + "rep/", awvW_save,
                                   wv_path=wv_path)
            awvW_instance.process_and_save()
            awvW_space = awvW_instance.rep.value

            hpam_save = SaveLoad(rewrite=True)
            hyper_param = HyperParameter(awvW_space, p_corpus.classes.value,class_names,
                                         [C_params, balance_params], ["C", "class_weight"], "LinearSVM", awvW_fn,
                                         output_folder, hpam_save, probability, rewrite_model=rewrite_all, folds=2)
            hyper_param.process_and_save()
            """
        if data_type != "movies":
            doc2vec_identifier =  "_" + str(dims[i]) + "_D2V"
            doc2vec_fn = file_name + doc2vec_identifier
            classify_doc2vec_fn = classifier_fn + doc2vec_identifier


            hpam_save = SaveLoad(rewrite=rewrite_all)

            hpam_dict["dim"] = [dims[i]]
            hpam_dict["corpus_fn"] = [p_corpus.processed_corpus.file_name]
            hpam_dict["wv_path"] = [wv_path_d2v]

            # Folds and space are determined inside of the method for this hyper-parameter selection, as it is stacked
            hyper_param = RecHParam(None, p_corpus.classes.value, class_names,  hpam_dict, kfold_hpam_dict, "d2v", model_type,
                                         doc2vec_fn, classify_doc2vec_fn, output_folder, hpam_save, probability=probability, rewrite_model=rewrite_all, dev_percent=dev_percent,
                                    data_type=data_type, score_metric=score_metric, auroc=auroc, matched_ids=matched_ids)
            hyper_param.process_and_save()
            all_test_result_rows.append(hyper_param.top_scoring_row_data.value)


        # Make the combined csv of all space types
    # Make the combined CSV of all the dims of all the space types
    all_r = np.asarray(all_test_result_rows).transpose()
    rows = all_r[1]
    cols = np.asarray(rows.tolist()).transpose()
    col_names = all_r[0][0]
    key = all_r[2]
    dt.write_csv(output_folder + "rep/score/csv_final/" +file_name+"reps"+model_type+".csv", col_names, cols, key)
    print("a")


def main(data_type, raw_folder, processed_folder,proj_folder="",  grams=0, model_type="LinearSVM", no_below=0.001,
         no_above=0.95, classes_freq_cutoff=100, bowmin=2, dev_percent=0.2, score_metric="avg_f1", max_depth=None):
    corpus_fn = ""
    if data_type == "newsgroups":
        newsgroups = fetch_20newsgroups(subset='all', shuffle=False, remove=("headers", "footers", "quotes"))
        corpus = newsgroups.data
        classes = newsgroups.target
        class_names = newsgroups.target_names
        name_of_class = "Newsgroups"
    elif data_type == "sentiment":
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=0, skip_top=0, index_from=0, seed=113)
        corpus = np.asarray(np.concatenate((x_train, x_test), axis=0))
        classes = np.asarray(np.concatenate((y_train, y_test), axis=0))
        corpus = np.asarray(process_corpus.makeCorpusFromIds(corpus, imdb.get_word_index()))
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
        classes_freq_cutoff = 100
        corpus = dt.import1dArray(raw_folder + "duplicate_removed_docs.txt")
        classes = dt.import2dArray(raw_folder + "unique_classes.txt", "i")
        class_names = dt.import1dArray(raw_folder + "class_names.txt")
        name_of_class = "Reuters"

    window_size = [1, 5, 15]
    min_count = [1, 5, 20]
    train_epoch = [20, 40, 100]

    dims = [20, 50, 100, 200]
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
    pipeline_fn = "num_stw"
    if data_type == "movies" or data_type == "placetypes":
        for i in range(len(classes)):
            pipeline(corpus, classes[i], class_names[i], pipeline_fn, processed_folder, dims, kfold_hpam_dict, hpam_dict, bowmin,
                 no_below,
                 no_above, classes_freq_cutoff, model_type, dev_percent, rewrite_all=False, remove_stop_words=True,
                 score_metric=score_metric, auroc=False, corpus_fn=corpus_fn, name_of_class=name_of_class[i])
    else:
        pipeline(corpus, classes, class_names, pipeline_fn, processed_folder, dims, kfold_hpam_dict, hpam_dict, bowmin, no_below,
             no_above, classes_freq_cutoff, model_type, dev_percent, rewrite_all=False, remove_stop_words=True, score_metric=score_metric, auroc=False,
                 corpus_fn=corpus_fn, name_of_class=name_of_class)
max_depths = [None, None, 3, 2, 1]
classifiers = ["LinearSVM", "DecisionTreeNone", "DecisionTree3", "DecisionTree2", "DecisionTree1"]
data_type = "movies"
if __name__ == '__main__':
    for i in range(len(classifiers)):
        main(data_type, "../../data/raw/"+data_type+"/",  "../../data/processed/"+data_type+"/", proj_folder="../../data/proj/"+data_type+"/",
                                grams=0, model_type=classifiers[i], no_below=0.001, no_above=0.95, classes_freq_cutoff=100, bowmin=2, dev_percent=0.2,
                                score_metric="avg_f1", max_depth=max_depths[i])