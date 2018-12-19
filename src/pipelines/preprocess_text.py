
from gensim.corpora import Dictionary
from gensim.utils import deaccent
from util import io as dt
from gensim.models.phrases import Phraser
from nltk.corpus import stopwords
import numpy as np
from keras.utils import to_categorical
import string
from sklearn.datasets import fetch_20newsgroups
import re
from os.path import expanduser
from keras.datasets import imdb
from sklearn.feature_extraction.text import CountVectorizer
from gensim.matutils import corpus2csc
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from test_representations import testAll
from model import ppmi
from util import py
from nltk.corpus import reuters
from model import pca
from collections import defaultdict
#import nltk
#nltk.download()
from data import process_corpus


# The overarching pipeline to obtain all prerequisite data for the derrac pipeline
# Todo: Better standaradize the saving/loading
def pipeline(corpus, classes, file_name, output_folder, bowmin, no_below, no_above, remove_stop_words=False):

    # Process and save corpus
    p_corpus = process_corpus.Corpus(corpus, classes, file_name, output_folder, bowmin, no_below, no_above, remove_stop_words)
    p_corpus.process_and_save()
    print("corpus done")

    filtered_bow = p_corpus.filtered_bow
    bow = p_corpus.bow

    # Get the PPMI values
    filtered_bow = filtered_bow.transpose()
    ppmi_matrix = ppmi.convertPPMISparse(filtered_bow)
    filtered_ppmi_sparse = sp.csr_matrix(ppmi_matrix).transpose()
    ppmi_matrix = ppmi.convertPPMISparse(bow)
    ppmi_sparse = sp.csr_matrix(ppmi_matrix).transpose()

    # Save the PPMI values
    sp.save_npz(output_folder + "bow/" + file_name + str(no_below) + "_" + str(no_above) + "_filtered_ppmi.npz", filtered_ppmi_sparse)
    sp.save_npz(output_folder + "bow/" + file_name + "_ppmi.npz", ppmi_sparse)

    dims = [20,50,100,200]

    standard_fn = output_folder + "rep/"

    for dim in dims:
        # Get the PCA space
        pca = pca.getPCA(pca, dim)

        # Save the PCA space
        np.save(pca, file_name + "_PCA_" + str(dim) + ".npy")

        # Get the AWV space

        # Save the AWV space



def main(data_type, raw_folder, processed_folder,proj_folder,  grams,  no_below, no_above, bowmin):
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
        classes = dt.import2dArray(raw_folder + "duplicate_removed_classes.txt", "i")

    # Run a pipeline that retains numbers and removes stopwords
    pipeline(corpus, classes, "num_stw", processed_folder, bowmin, no_below, no_above, remove_stop_words=True)




data_type = "reuters"
if __name__ == '__main__': main(data_type, "../../data/raw/"+data_type+"/",  "../../data/processed/"+data_type+"/", "../../data/proj/"+data_type+"/",  0, 10, 0.95, 2)