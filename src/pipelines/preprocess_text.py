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
from pipelines.KFoldHyperParameter import KFoldHyperParameter
from pipelines.KFoldHyperParameter import HyperParameter

# The overarching pipeline to obtain all prerequisite data for the derrac pipeline
# Todo: Better standaradize the saving/loading
def pipeline(corpus, classes, class_names, file_name, output_folder, dims, kfold_hpam_dict, hpam_dict, bowmin, no_below_fraction, no_above, classes_freq_cutoff, model_type, rewrite_all=False, remove_stop_words=False,  auroc=False):

    probability = False
    if auroc is True:
        probability = True

    doc_amt = split.get_doc_amt(data_type)
    no_below = int(doc_amt * no_below_fraction)
    print("Filtering all words that do not appear in", no_below, "documents")

    # Process and save corpus
    corpus_save = SaveLoad(rewrite=True)
    p_corpus = process_corpus.Corpus(corpus, classes, class_names, file_name, output_folder, bowmin, no_below, no_above, classes_freq_cutoff, remove_stop_words, corpus_save)
    p_corpus.process_and_save()

    # Get the PPMI values
    ppmi_save = SaveLoad(rewrite=rewrite_all)
    ppmi_fn = file_name + "_ppmi"
    ppmi_unfiltered = ppmi.PPMI(p_corpus.bow.value, doc_amt, output_folder + "bow/" + ppmi_fn, ppmi_save)
    ppmi_unfiltered.process_and_save()
    ppmi_unf_matrix = ppmi_unfiltered.ppmi_matrix.value

    """
    ppmi_save = SaveLoad(rewrite=rewrite_all)
    ppmi_filtered = ppmi.PPMI(p_corpus.filtered_bow.value, doc_amt, output_folder + "bow/NB_" + str(no_below) + "_NA_" + str(no_above) + ppmi_fn, ppmi_save)
    ppmi_filtered.process_and_save()
    ppmi_filtered_matrix = ppmi_filtered.ppmi_matrix.value
    
    hpam_save = SaveLoad(rewrite=rewrite_all)
    hyper_param = KFoldHyperParameter(ppmi_filtered_matrix.toarray(), p_corpus.classes.value, p_corpus.filtered_class_names.value,
                                      kfold_hpam_dict, model_type, ppmi_fn,
                                      output_folder, hpam_save, probability, rewrite_model=rewrite_all, folds=2)
    hyper_param.process_and_save()
    """
    # Creating and testing spaces, MDS not included in the creation process
    for i in range(len(dims)):
        pca_save = SaveLoad(rewrite=rewrite_all)
        pca_fn = file_name + "_" + str(dims[i]) + "_PCA"
        pca_instance = pca.PCA(ppmi_unf_matrix, doc_amt, dims[i],
                              pca_fn,  output_folder + "rep/pca/", pca_save)
        pca_instance.process_and_save()
        pca_space = pca_instance.rep.value

        hpam_save = SaveLoad(rewrite=rewrite_all)
        hyper_param = KFoldHyperParameter(pca_space, p_corpus.filtered_classes.value, p_corpus.filtered_class_names.value,
                                          kfold_hpam_dict, model_type, pca_fn,
                                     output_folder, hpam_save, probability, rewrite_model=rewrite_all, folds=2)
        hyper_param.process_and_save()

        wv_path = "D:/PhD/Code/ThesisPipeline/ThesisPipeline/data/raw/glove/glove.6B." + str(dims[i]) + 'd.txt'
        wv_path_d2v = "D:/PhD/Code/ThesisPipeline/ThesisPipeline/data/raw/glove/glove.6B.300d.txt"


        # We only have word-vectors of size 50, 100, 200 and 300
        if dims[i] != 20:
            awv_save = SaveLoad(rewrite=rewrite_all)
            awv_fn = file_name + "_" + str(dims[i]) + "_AWV"

            awv_instance = awv.AWV(p_corpus.split_corpus.value, dims[i], awv_fn, output_folder + "rep/awv/" , awv_save, wv_path=wv_path)
            awv_instance.process_and_save()
            awv_space = awv_instance.rep.value

            hpam_save = SaveLoad(rewrite=rewrite_all)
            hyper_param = KFoldHyperParameter(awv_space, p_corpus.filtered_classes.value, p_corpus.filtered_class_names.value,
                                              kfold_hpam_dict, model_type, awv_fn,
                                         output_folder, hpam_save, probability, rewrite_model=rewrite_all, folds=2)
            hyper_param.process_and_save()
            """
            awvW_save = SaveLoad(rewrite=rewrite_all)
            awvW_fn = file_name + "_" + str(dims[i]) + "_awvW"

            awvW_instance = awv.AWVw(ppmi_unf_matrix, p_corpus.id2token.value, dims[i], awvW_fn, output_folder + "rep/", awvW_save,
                                   wv_path=wv_path)
            awvW_instance.process_and_save()
            awvW_space = awvW_instance.rep.value

            hpam_save = SaveLoad(rewrite=True)
            hyper_param = HyperParameter(awvW_space, p_corpus.classes.value, p_corpus.filtered_class_names.value,
                                         [C_params, balance_params], ["C", "class_weight"], "LinearSVM", awvW_fn,
                                         output_folder, hpam_save, probability, rewrite_model=rewrite_all, folds=2)
            hyper_param.process_and_save()
            """

        doc2vec_fn = file_name + "_" + str(dims[i]) + "_D2V"


        hpam_save = SaveLoad(rewrite=rewrite_all)

        hpam_dict["dim"] = [dims[i]]
        hpam_dict["corpus_fn"] = [p_corpus.processed_corpus.file_name]
        hpam_dict["wv_path"] = [wv_path_d2v]

        # Folds and space are determined inside of the method for this hyper-parameter selection, as it is stacked
        hyper_param = HyperParameter(None, p_corpus.filtered_classes.value, p_corpus.filtered_class_names.value,  hpam_dict, kfold_hpam_dict, "d2v", model_type,
                                     doc2vec_fn, output_folder, hpam_save, probability, rewrite_model=rewrite_all)
        hyper_param.process_and_save()

        # Include the MDS imports and test them in the same way


        # Make the combined csv of all space types
    # Make the combined CSV of all the dims of all the space types



def main(data_type, raw_folder, processed_folder,proj_folder,  grams, model_type, no_below, no_above, classes_freq_cutoff, bowmin):
    if data_type == "newsgroups":
        newsgroups = fetch_20newsgroups(subset='all', shuffle=False, remove=("headers", "footers", "quotes"))
        corpus = newsgroups.data
        classes = newsgroups.target
        class_names = dt.import1dArray(raw_folder + "class_names.txt")
    elif data_type == "sentiment":
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=0, skip_top=0, index_from=0, seed=113)
        corpus = np.concatenate((x_train, x_test), axis=0)
        classes = np.concatenate((y_train, y_test), axis=0)
        corpus = process_corpus.makeCorpusFromIds(corpus, imdb.get_word_index())
        class_names = ["sentiment"]
        classes_freq_cutoff = 0
    else:
        classes_freq_cutoff = 100
        corpus = dt.import1dArray(raw_folder + "duplicate_removed_docs.txt")
        classes = dt.import2dArray(raw_folder + "unique_classes.txt", "i")
        class_names = dt.import1dArray(raw_folder + "class_names.txt")

    window_size = [1, 5, 15]
    min_count = [1, 5, 20]
    train_epoch = [20, 40, 100]

    dims = [20, 50, 100, 200]
    balance_params = ["balanced", None]
    C_params = [1.0, 0.01, 0.001, 0.0001]
    gamma_params = [1.0, 0.01, 0.001, 0.0001]

    n_estimators = [ 1000, 2000]
    max_features = ['auto', 'log2']
    criterion = ["gini"]
    max_depth = [None]
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


    hpam_dict = { "window_size":window_size,
                "min_count":min_count,
                "train_epoch":train_epoch}

    pipeline(corpus, classes, class_names, "num_stw", processed_folder, dims, kfold_hpam_dict, hpam_dict, bowmin, no_below, no_above, classes_freq_cutoff, model_type, rewrite_all=False, remove_stop_words=True)




data_type = "newsgroups"
if __name__ == '__main__': main(data_type, "../../data/raw/"+data_type+"/",  "../../data/processed/"+data_type+"/", "../../data/proj/"+data_type+"/",  0, "LinearSVM", 0.01, 0.95, 100, 2)