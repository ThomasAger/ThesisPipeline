from util import io as dt
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from keras.datasets import imdb
from rep import pca, ppmi
#import nltk
#nltk.download()
from data import process_corpus
from util.save_load import SaveLoad
from util import split
from pipelines.KFoldHyperParameter import HyperParameter


# The overarching pipeline to obtain all prerequisite data for the derrac pipeline
# Todo: Better standaradize the saving/loading
def pipeline(corpus, classes, class_names, file_name, output_folder, bowmin, no_below, no_above, classes_freq_cutoff, rewrite_all=False, remove_stop_words=False,  auroc=True):

    probability = False
    if auroc is True:
        probability = True

    doc_amt = split.get_doc_amt(data_type)

    # Process and save corpus
    corpus_save = SaveLoad(rewrite=rewrite_all)
    p_corpus = process_corpus.Corpus(corpus, classes, class_names, file_name, output_folder, bowmin, no_below, no_above, classes_freq_cutoff, remove_stop_words, corpus_save)
    p_corpus.process_and_save()

    # Get the PPMI values
    ppmi_save = SaveLoad(rewrite=rewrite_all)
    ppmi_unfiltered = ppmi.PPMI(p_corpus.bow.value, doc_amt, output_folder + "bow/" + file_name + "_", ppmi_save)
    ppmi_unfiltered.process_and_save()
    ppmi_unf_matrix = ppmi_unfiltered.ppmi_matrix.value
    """
    # Get the PPMI values
    ppmi_filtered_save = SaveLoad()
    ppmi_filtered = ppmi.PPMI(p_corpus.filtered_bow.value, doc_amt,  output_folder + "bow/" + str(no_above) + "_" + str(no_below) + "_" + file_name + "_", ppmi_filtered_save)
    ppmi_filtered.process_and_save()
    ppmi_filtered_matrix = ppmi_filtered.ppmi_matrix.value
    """
    dims = [20, 50, 100, 200]
    balance_params = ["balanced", None]
    C_params = [1.0, 0.01, 0.001, 0.0001]
    for i in range(len(dims)):
        pca_save = SaveLoad(rewrite=rewrite_all)
        pca_fn = file_name + "_" + str(dims[i]) + "_PCA"
        pca_instance = pca.PCA(ppmi_unf_matrix, doc_amt, dims[i],
                               output_folder + "rep/" + pca_fn, pca_save)
        pca_instance.process_and_save()
        pca_space = pca_instance.PCA.value
        
        
        awv_save = SaveLoad(rewrite=rewrite_all)
        awv_fn = file_name + "_" + str(dims[i]) + "_AWV"

        awv_instance = pca.PCA(ppmi_unf_matrix, doc_amt, dims[i],
                               output_folder + "rep/" + pca_fn, pca_save)
        awv_instance.process_and_save()
        awv_space = awv_instance.PCA.value

        awvW_save = SaveLoad(rewrite=rewrite_all)
        awvW_fn = file_name + "_" + str(dims[i]) + "_awvW"

        awvW_instance = pca.PCA(ppmi_unf_matrix, doc_amt, dims[i],
                               output_folder + "rep/" + pca_fn, pca_save)
        awvW_instance.process_and_save()
        awvW_space = awvW_instance.PCA.value

        doc2vec_save = SaveLoad(rewrite=rewrite_all)
        doc2vec_fn = file_name + "_" + str(dims[i]) + "_doc2vec"

        doc2vec_instance = pca.PCA(ppmi_unf_matrix, doc_amt, dims[i],
                                output_folder + "rep/" + pca_fn, pca_save)
        doc2vec_instance.process_and_save()
        doc2vec_space = doc2vec_instance.PCA.value
        
        # Make the combined csv of all spaces
        hpam_save = SaveLoad(rewrite=True)
        hyper_param = HyperParameter(pca_space, p_corpus.classes.value, p_corpus.filtered_class_names.value,
                                     [C_params, balance_params], ["C", "class_weight"], "LinearSVM", pca_fn,
                                     output_folder, hpam_save, probability, folds=2)
        hyper_param.process_and_save()



def main(data_type, raw_folder, processed_folder,proj_folder,  grams,  no_below, no_above, classes_freq_cutoff, bowmin):
    if data_type == "newsgroups":
        newsgroups = fetch_20newsgroups(subset='all', shuffle=False, remove=("headers", "footers", "quotes"))
        corpus = newsgroups.data
        classes = newsgroups.target
    elif data_type == "sentiment":
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=0, skip_top=0, index_from=0, seed=113)
        corpus = np.concatenate((x_train, x_test), axis=0)
        classes = np.concatenate((y_train, y_test), axis=0)
        corpus = process_corpus.makeCorpusFromIds(corpus, imdb.get_word_index())
    else:
        corpus = dt.import1dArray(raw_folder + "duplicate_removed_docs.txt")
        classes = dt.import2dArray(raw_folder + "unique_classes.txt", "i")
        class_names = dt.import1dArray(raw_folder + "class_names.txt")

    # Run a pipeline that retains numbers and removes stopwords
    pipeline(corpus, classes, class_names, "num_stw", processed_folder, bowmin, no_below, no_above, classes_freq_cutoff, rewrite_all=False, remove_stop_words=True)




data_type = "reuters"
if __name__ == '__main__': main(data_type, "../../data/raw/"+data_type+"/",  "../../data/processed/"+data_type+"/", "../../data/proj/"+data_type+"/",  0, 10, 0.95, 100, 2)