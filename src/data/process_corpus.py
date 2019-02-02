from gensim.corpora import Dictionary
from gensim.utils import deaccent
from gensim.models.phrases import Phraser
from nltk.corpus import stopwords
from common import Method
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
from nltk.corpus import reuters
from collections import defaultdict
from common.SaveLoadPOPO import SaveLoadPOPO
from util import proj
from util import py
from util import py
# Has batch processing (haven't figured out how to use it yet)
# Retains punctuation e.g. won't -> "won't"
# Has many additional options
# Seems hard to get vocab, tokens (maybe misunderstanding)
# Retains brackets, etc , e.g. ["(i", "think", "so)"]
# Can make vocab from your own corpus, but has different quirks
def spacyTokenize(corpus):  # Documentation unclear
    tokenized_corpus = np.empty(len(corpus), dtype=np.object)
    tokenized_ids = np.empty(len(corpus), dtype=np.object)
    processed_corpus = np.empty(len(corpus), dtype=np.object)
    for i in range(len(corpus)):
        corpus[i] = corpus[i].replace("\n", " ")
    # vocab_spacy = Vocab(strings=corpus)
    tokenizer_spacy = Tokenizer(nlp.vocab)
    for i in range(len(corpus)):
        spacy_sent = tokenizer_spacy(corpus[i])
        processed_corpus[i] = spacy_sent.text
        tokenized_corpus[i] = list(spacy_sent)
        for j in range(len(tokenized_corpus[i])):
            tokenized_corpus[i][j] = tokenized_corpus[i][j].text
        tokenized_ids[i] = spacy_sent.to_array([spacy.attrs.ID])
        sd = 0
    return processed_corpus, tokenized_corpus, tokenized_ids, [None]


def tokenizeNLTK1(corpus):  # This is prob better than current implementation - look into later
    processed_corpus = np.empty(len(corpus), dtype=np.object)
    tokenized_corpus = np.empty(len(corpus), dtype=np.object)
    tokenized_ids = np.empty(len(corpus), dtype=np.object)
    for i in range(len(corpus)):
        i = 0
    return processed_corpus, tokenized_corpus, tokenized_ids


def tokenizeNLTK2(corpus):
    processed_corpus = np.empty(len(corpus), dtype=np.object)
    tokenized_corpus = np.empty(len(corpus), dtype=np.object)
    tokenized_ids = np.empty(len(corpus), dtype=np.object)
    return processed_corpus, tokenized_corpus, tokenized_ids


# PAT_ALPHABETIC = re.compile(r'(((?![\d])\w)+)', re.UNICODE) # stolen from gensim
PAT_ALPHANUMERIC = re.compile(r'((\w)+)', re.UNICODE)  # stolen from gensim, but added numbers included


def tokenize(text):
    for match in PAT_ALPHANUMERIC.finditer(text):
        yield match.group()


# Removes punctuation "won't be done, dude-man." = ["wont", "be", "done", "dudeman"]
# Lowercase and deaccenting "cYkět" = ["cyket"]
# Converting to ID's requires separate process using those vocabs. More time
# Finds phrases using gensim, e.g. "mayor" "of" "new" "york" -> "mayor" "of" "new_york"
def naiveTokenizer(corpus):
    tokenized_corpus = np.empty(len(corpus), dtype=np.object)
    for i in range(len(corpus)):
        tokenized_corpus[i] = list(tokenize(corpus[i]))
        for j in reversed(range(len(tokenized_corpus[i]))):
            if len(tokenized_corpus[i][j]) == 1:
                del tokenized_corpus[i][j]
    return tokenized_corpus


def getVocab(tokenized_corpus):
    dct = Dictionary(tokenized_corpus)
    vocab = dct.token2id
    # id2token needs to be used before it can be obtained, so here we use it arbitrarily
    _ = dct[0]
    id2token = dct.id2token
    return vocab, dct, id2token



def doc2bow(tokenized_corpus, dct, bowmin):
    dct.filter_extremes(no_below=bowmin)  # Most occur in at least 2 documents
    bow = [dct.doc2bow(text) for text in tokenized_corpus]
    bow = corpus2csc(bow)
    vocab = dct.token2id
    return dct, bow, vocab


def filterBow_sklearn(processed_corpus, no_below, no_above):  # sklearn is slightly worse here
    tf_vectorizer = CountVectorizer(max_df=no_above, min_df=no_below, stop_words=None)
    print("completed vectorizer")
    tf = tf_vectorizer.fit(processed_corpus)
    feature_names = tf.get_feature_names()
    tf = tf_vectorizer.transform(processed_corpus)
    return tf.transpose(), feature_names


def filterBow(tokenized_corpus, dct, no_below, no_above):
    dct.filter_extremes(no_below=no_below, no_above=no_above)
    filtered_bow = [dct.doc2bow(text) for text in tokenized_corpus]
    filtered_bow = corpus2csc(filtered_bow)
    filtered_vocab = dct.token2id
    return filtered_bow, list(dct.token2id.keys()), filtered_vocab



def removeEmpty(processed_corpus, tokenized_corpus, classes):
    remove_ind = []
    for i in range(len(processed_corpus)):
        if len(tokenized_corpus[i]) < 0:
            print("DEL", processed_corpus[i])
            remove_ind.append(i)
    processed_corpus = np.delete(processed_corpus, remove_ind)
    tokenized_corpus = np.delete(tokenized_corpus, remove_ind)
    classes = np.delete(classes, remove_ind, axis=0)
    return processed_corpus, tokenized_corpus, remove_ind, classes


def preprocess(corpus):
    preprocessed_corpus = np.empty(len(corpus), dtype=np.object)

    # Remove new line characters
    table = str.maketrans(dict.fromkeys("\n\r", " "))
    for i in range(len(preprocessed_corpus)):
        preprocessed_corpus[i] = corpus[i].translate(table)

    # Table for punctuation
    table = str.maketrans(dict.fromkeys(string.punctuation))
    for i in range(len(corpus)):
        # Lowercase
        preprocessed_corpus[i] = preprocessed_corpus[i].lower()
        # Remove all punctuation
        preprocessed_corpus[i] = preprocessed_corpus[i].translate(table)
        # Replace all whitespace with single whitespace
        preprocessed_corpus[i] = re.sub(r'\s+', ' ', preprocessed_corpus[i])
        # Deaccent
        preprocessed_corpus[i] = deaccent(preprocessed_corpus[i])
        # Strip trailing whitespace
        preprocessed_corpus[i] = preprocessed_corpus[i].strip()

    return preprocessed_corpus


def removeStopWords(tokenized_corpus):
    new_tokenized_corpus = np.empty(len(tokenized_corpus), dtype=np.object)
    stop_words_corpus = np.empty(len(tokenized_corpus), dtype=np.object)
    stop_words = set(stopwords.words('english'))
    for i in range(len(tokenized_corpus)):
        new_tokenized_corpus[i] = [w for w in tokenized_corpus[i] if w not in stop_words]
        stop_words_corpus[i] = " ".join(new_tokenized_corpus[i])
    return new_tokenized_corpus, stop_words_corpus


def tokensToIds(tokenized_corpus, vocab):
    tokenized_ids = np.empty(len(tokenized_corpus), dtype=np.object)
    for i in range(len(tokenized_corpus)):
        ids = np.empty(len(tokenized_corpus[i]), dtype=np.object)
        for t in range(len(tokenized_corpus[i])):
            ids[t] = vocab[tokenized_corpus[i][t]]
        tokenized_ids[i] = ids
    return tokenized_ids


# This causes OOM error. Need to rework
def ngrams(tokenized_corpus):  # Increase the gram amount by 1
    processed_corpus = np.empty(len(tokenized_corpus), dtype=np.object)
    phrases = Phrases(tokenized_corpus)
    gram = Phraser(phrases)
    for i in range(len(tokenized_corpus)):
        tokenized_corpus[i] = gram[tokenized_corpus[i]]
        processed_corpus[i] = " ".join(tokenized_corpus[i])
    return processed_corpus, tokenized_corpus


def averageWV(tokenized_corpus, depth):
    print("")


def averageWVPPMI(tokenized_corpus, ppmi):
    print("")


# For sentiment etc
def makeCorpusFromIds(tokenized_ids, vocab):
    vocab = {k: (v + 0) for k, v in vocab.items()}
    vocab["<UNK>"] = 0
    vocab["<START>"] = 1
    vocab["<OOV>"] = 2
    id_to_word = {value: key for key, value in vocab.items()}

    processed_corpus = np.empty(shape=(len(tokenized_ids)), dtype=np.object)  # Have to recreate original word vectors
    for s in range(len(tokenized_ids)):
        word_sentence = []
        for w in range(len(tokenized_ids[s])):
            word_sentence.append(id_to_word[tokenized_ids[s][w]])
        processed_corpus[s] = " ".join(word_sentence)

    return processed_corpus



def split_all(corpus):
    split_corpus = []
    for i in range(len(corpus)):
        split_corpus.append(corpus[i].split())
    return split_corpus

class LimitWords(Method.Method):

    word_list = None
    no_below = None
    no_above = None
    output_folder = None
    bow_word_dict = None
    dct = None
    bow = None

    def __init__(self, file_name, save_class, dct, bow, output_folder, word_list, no_below, no_above):
        self.word_list = word_list
        self.no_above = no_above
        self.no_below = no_below
        self.output_folder = output_folder
        self.dct = dct
        self.bow = bow
        super().__init__(file_name, save_class)

    def getBowWordDct(self):
        if self.bow_word_dict.value is None:
            self.bow_word_dict.value = self.save_class.load(self.bow_word_dict)
        return self.bow_word_dict.value

    def getNewWordDict(self):
        if self.new_word_dict.value is None:
            self.new_word_dict.value = self.save_class.load(self.new_word_dict)
        return self.new_word_dict.value

    def makePopos(self):
        self.bow_word_dict = SaveLoadPOPO(None, self.output_folder + self.file_name + "_wldct_NB_" + str(self.no_below) + "_NA_" + str(self.no_above), "npy_dict")
        self.new_word_dict = SaveLoadPOPO(None, self.output_folder + self.file_name + "_new_wdct_NB_" + str(self.no_below) + "_NA_" + str(self.no_above), "npy_dict")


    def makePopoArray(self):
        self.popo_array = [self.bow_word_dict, self.new_word_dict]

    def process(self):
        orig_dct = self.dct.token2id
        self.dct.filter_extremes(no_below=self.no_below, no_above=self.no_above)
        self.bow_word_dict.value = dict(self.dct.token2id)
        self.new_word_dict.value = dict(self.dct.token2id)
        for key, value in self.bow_word_dict.value.items():
            self.bow_word_dict.value[key] = orig_dct[key]
        super().process()

class MasterCorpus(Method.Method):
    orig_classes = None
    name_of_class = None
    tokenized_corpus = None
    tokenized_ids = None
    id2token = None
    bow = None
    bow_vocab = None
    filtered_vocab = None
    processed_corpus = None
    filtered_bow = None
    word_list = None
    dct = None
    remove_ind = None
    classes_categorical = None
    all_words = None
    output_folder = None
    bowmin = None
    no_below = None
    no_above = None
    file_dict = None
    loaded_file_dict = None
    all_vocab = None
    corpus = None
    classes = None
    filtered_class_names = None
    filtered_classes = None
    split_corpus = None
    bow_dict = None

    def __init__(self, orig_classes, name_of_class, file_name, output_folder, bowmin, no_below,
                 no_above, remove_stop_words, save_class):
        self.orig_classes = orig_classes
        self.output_folder = output_folder
        self.bowmin = bowmin
        self.no_below = no_below
        self.name_of_class = name_of_class
        self.no_above = no_above
        self.remove_stop_words = remove_stop_words
        self.output_folder = output_folder
        self.file_name = file_name
        super().__init__(file_name, save_class)

    def getSplitCorpus(self):
        return self.save_class.load(self.split_corpus)
    def getClasses(self):
        return self.save_class.load(self.classes)
    def getBow(self):
        return self.save_class.load(self.bow)
    def getFilteredBow(self):
        return self.save_class.load(self.filtered_bow)

    def getDct(self):
        self.dct =  self.save_class.load(self.dct)
        return self.dct

    def getAllWords(self):
        self.all_words =  self.save_class.load(self.all_words)
        return self.all_words

class Corpus(MasterCorpus):
    orig_corpus = None

    def __init__(self, orig_corpus, orig_classes, name_of_class, file_name, output_folder, bowmin, no_below, no_above,
                 remove_stop_words, save_class):
        self.orig_corpus = orig_corpus
        super().__init__(orig_classes, name_of_class, file_name, output_folder, bowmin, no_below,
                 no_above,  remove_stop_words, save_class)

    def makePopos(self):
        output_folder = self.output_folder
        file_name = self.file_name
        standard_fn = output_folder + "bow/"
        self.dct = SaveLoadPOPO(self.dct, standard_fn + "metadata/" + file_name + ".pkl", "gensim")
        self.bowdict = SaveLoadPOPO(self.dct, standard_fn + "metadata/" + file_name + "_bowdict.pkl", "gensim")
        self.remove_ind = SaveLoadPOPO(self.remove_ind, standard_fn + "metadata/" + file_name + "_remove.npy", "npy")
        self.tokenized_corpus = SaveLoadPOPO(self.tokenized_corpus, standard_fn + file_name + "_tokenized_corpus.npy", "npy")
        self.tokenized_ids = SaveLoadPOPO(self.tokenized_ids, standard_fn + file_name + "_tokenized_ids.npy", "npy")
        self.id2token = SaveLoadPOPO(self.id2token, standard_fn + "metadata/" + file_name + "id2token.npy", "npy_dict")
        self.all_vocab = SaveLoadPOPO(self.all_vocab, standard_fn + "metadata/" + file_name + "_all_vocab.npy", "npy_dict")
        self.bow_vocab = SaveLoadPOPO(self.bow_vocab,
                                      standard_fn + "metadata/" + file_name + "_vocab_" + str(self.bowmin) + ".npy",
                                      "npy")
        self.filtered_vocab = SaveLoadPOPO(self.filtered_vocab,
                                           standard_fn + "metadata/" + file_name + "_filtered_vocab.npy", "npy")
        self.processed_corpus = SaveLoadPOPO(self.processed_corpus,
                                             output_folder + "corpus/" + file_name + "_corpus_processed.npy", "npy")
        self.split_corpus = SaveLoadPOPO(self.split_corpus,
                                             output_folder + "corpus/" + file_name + "_corpus_processed_split.npy", "npy")
        print(self.name_of_class )
        self.classes = SaveLoadPOPO(self.classes, output_folder + "classes/" + file_name + self.name_of_class +"_classes.npy", "npy")

        self.bow = SaveLoadPOPO(self.bow, standard_fn + file_name + "_sparse_corpus.npz", "scipy")
        self.filtered_bow = SaveLoadPOPO(self.filtered_bow,
                                         standard_fn + file_name + "_" + str(self.no_below) + "_" + str(
                                             self.no_above) + "_filtered.npz", "scipy")
        self.word_list = SaveLoadPOPO(self.word_list, standard_fn + "metadata/" + file_name + "_words.npy", "npy")
        self.all_words = SaveLoadPOPO(self.all_words, standard_fn + "metadata/" + file_name + "_all_words_2.npy",
                                      "npy")
    def getCorpus(self):
        self.processed_corpus = self.save_class.load(self.processed_corpus)
        return self.processed_corpus


    def makePopoArray(self):
        self.popo_array = [self.dct, self.remove_ind, self.tokenized_corpus, self.tokenized_ids,
                           self.id2token,
                           self.bow_vocab, self.filtered_vocab,
                           self.processed_corpus, self.classes,  self.bow, self.filtered_bow,
                           self.word_list, self.all_words,  self.split_corpus, self.bowdict]

    def process(self):
        print("Original doc len", len(self.orig_corpus))
        self.processed_corpus.value = preprocess(self.orig_corpus)
        self.tokenized_corpus.value = naiveTokenizer(self.processed_corpus.value)
        if self.remove_stop_words:
            self.tokenized_corpus.value, self.processed_corpus.value = removeStopWords(self.tokenized_corpus.value)
        self.processed_corpus.value, self.tokenized_corpus.value, self.remove_ind.value, self.classes.value = removeEmpty(self.processed_corpus.value,
                                                                                                  self.tokenized_corpus.value,self.orig_classes)

        self.split_corpus.value = split_all(self.processed_corpus.value)

        self.all_vocab.value, self.dct.value, self.id2token.value = getVocab(self.tokenized_corpus.value)
        self.bowdict.value, self.bow.value, self.bow_vocab.value = doc2bow(self.tokenized_corpus.value, self.dct.value, self.bowmin)
        self.all_words.value = list(self.bow_vocab.value.keys())
        print(self.bowmin, len(self.all_words.value), "|||", self.bow.value.shape)
        self.filtered_bow.value, self.word_list.value, self.filtered_vocab.value = filterBow(self.tokenized_corpus.value, self.bowdict.value,
                                                                           self.no_below, self.no_above)
        # The idea here was to clear out anything empty after the big filtering, but it's probably fine.
        #self.filtered_bow.value, self.filtered_classes.value = removeEmptyBow(self.filtered_bow.value, self.classes.value,self.orig_classes)
        self.tokenized_ids.value = tokensToIds(self.tokenized_corpus.value, self.all_vocab.value)

        super().process()

# Does not support basic tokenization or clean up methods yet, only gensim methods
class StreamedCorpus(MasterCorpus):
    corpus_fn_to_stream = None
    def __init__(self,  orig_classes, name_of_class, file_name, output_folder, bowmin, no_below,
                 no_above,
                 remove_stop_words, save_class, corpus_fn_to_stream=None):
        self.corpus_fn_to_stream = corpus_fn_to_stream
        super().__init__(orig_classes, name_of_class, file_name, output_folder, bowmin, no_below,
                 no_above,remove_stop_words, save_class)

    def makePopos(self):
        output_folder = self.output_folder
        file_name = self.file_name
        standard_fn = output_folder + "bow/"
        self.dct = SaveLoadPOPO(self.dct, standard_fn + "metadata/" + file_name + ".pkl", "gensim")
        self.bowdict = SaveLoadPOPO(self.dct, standard_fn + "metadata/" + file_name + "_bowdict.pkl", "gensim")
        self.tokenized_corpus = SaveLoadPOPO(self.tokenized_corpus, standard_fn + file_name + "_tokenized_corpus.npy", "npy")
        self.tokenized_ids = SaveLoadPOPO(self.tokenized_ids, standard_fn + file_name + "_tokenized_ids.npy", "npy")
        self.id2token = SaveLoadPOPO(self.id2token, standard_fn + "metadata/" + file_name + "id2token.dct", "dct")
        self.all_vocab = SaveLoadPOPO(self.all_vocab, standard_fn + "metadata/" + file_name + "_all_vocab.dct", "dct")
        self.bow_vocab = SaveLoadPOPO(self.bow_vocab,
                                      standard_fn + "metadata/" + file_name + "_vocab_" + str(self.bowmin) + ".npy",
                                      "npy")
        self.filtered_vocab = SaveLoadPOPO(self.filtered_vocab,
                                           standard_fn + "metadata/" + file_name + "_filtered_vocab.npy", "npy")
        self.processed_corpus = SaveLoadPOPO(self.processed_corpus,
                                             output_folder + "corpus/" + file_name + "_corpus_processed.txt", "1dtxts")
        self.split_corpus = SaveLoadPOPO(self.split_corpus,
                                             output_folder + "corpus/" + file_name + "_corpus_processed_split.npy", "npy")
        self.classes = SaveLoadPOPO(self.classes, output_folder + "classes/" + file_name +self.name_of_class +  "_classes.npy", "npy")
        self.filtered_classes = SaveLoadPOPO(self.filtered_classes, output_folder + "classes/" + file_name +self.name_of_class +  "_fil_classes.npy", "npy")
        self.classes_categorical = SaveLoadPOPO(self.classes_categorical,
                                                output_folder + "classes/" + file_name + self.name_of_class + "_classes_categorical.npy",
                                                "npy")
        self.filtered_class_names = SaveLoadPOPO(self.filtered_class_names,
                                                output_folder + "classes/" + file_name +self.name_of_class +  "_class_names.txt",
                                                "1dtxts")
        self.bow = SaveLoadPOPO(self.bow, standard_fn + file_name + "_sparse_corpus.npz", "scipy")
        self.filtered_bow = SaveLoadPOPO(self.filtered_bow,
                                         standard_fn + file_name + "_" + str(self.no_below) + "_" + str(
                                             self.no_above) + "_filtered.npz", "scipy")
        self.word_list = SaveLoadPOPO(self.word_list, standard_fn + "metadata/" + file_name + "_words.txt", "1dtxts")
        self.all_words = SaveLoadPOPO(self.all_words, standard_fn + "metadata/" + file_name + "_all_words_2.txt",
                                      "1dtxts")

    def getBow(self):
        self.bow.value = self.save_class.load(self.bow)
        return self.bow.value

    def getWordList(self):
        self.all_words.value = self.save_class.load(self.all_words)
        return self.all_words.value

    def getFilteredBow(self):
        self.filtered_bow.value = self.save_class.load(self.filtered_bow)
        return self.filtered_bow.value

    def getFilteredWordList(self):
        self.word_list.value = self.save_class.load(self.word_list)
        return self.word_list.value


    def makePopoArray(self):
        self.popo_array = [self.dct,
                           self.id2token,
                           self.bow_vocab, self.filtered_vocab,
                            self.classes,  self.bow, self.filtered_bow,
                           self.word_list, self.all_words, self.bowdict]
        if self.classes_categorical.value is None:
            self.popo_array = [self.dct,
                               self.id2token,
                               self.bow_vocab, self.filtered_vocab,
                                self.classes, self.bow, self.filtered_bow,
                               self.word_list, self.all_words, self.bowdict]
    def process(self):
        self.classes.value = self.orig_classes
        self.all_vocab.value, self.dct.value, self.id2token.value = getVocabStreamed(self.corpus_fn_to_stream)
        self.bowdict.value, self.bow.value, self.bow_vocab.value = doc2bowStreamed(self.corpus_fn_to_stream, self.dct.value, self.bowmin)
        self.all_words.value = list(self.bowdict.value.keys())
        print(self.bowmin, len(self.all_words.value), "|||", self.bow.value.shape)
        self.filtered_bow.value, self.word_list.value, self.filtered_vocab.value = filterBowStreamed(
            self.corpus_fn_to_stream, self.bowdict.value,
            self.no_below, self.no_above)
        super().process()
        # We are not doing any changes to the classes/documents

class ProcessClasses(MasterCorpus):
    classes_freq_cutoff = None
    orig_class_names = None
    def __init__(self, orig_classes, orig_class_names, file_name, output_folder, bowmin, no_below,
                 no_above, classes_freq_cutoff, remove_stop_words, save_class, name_of_class):
        # If it's a multi class array
        if orig_classes is not None and py.isArray(orig_classes[0]) is True and np.amax(orig_classes[0]) == 1:
            orig_classes = py.transIfRowsLarger(orig_classes)
        self.classes_freq_cutoff = classes_freq_cutoff
        self.orig_class_names = orig_class_names
        super().__init__(orig_classes, name_of_class, file_name, output_folder, bowmin, no_below,no_above, remove_stop_words, save_class)

    def makePopos(self):
        output_folder = self.output_folder
        file_name = self.file_name
        standard_fn = output_folder + "bow/"

        self.filtered_classes = SaveLoadPOPO(self.filtered_classes, output_folder + "classes/" + file_name + self.name_of_class + "_fil_classes.npy", "npy")
        self.classes_categorical = SaveLoadPOPO(self.classes_categorical,
                                                output_folder + "classes/" + file_name + self.name_of_class + "_classes_categorical.npy",
                                                "npy")
        self.filtered_class_names = SaveLoadPOPO(self.filtered_class_names,
                                                output_folder + "classes/" + file_name + self.name_of_class + "_class_names.txt",
                                                "1dtxts")


    def makePopoArray(self):
        self.popo_array = [  self.filtered_class_names, self.filtered_classes, self.classes_categorical]

    def process(self):

        print("classes", len(self.orig_classes))
        self.classes_categorical.value = self.orig_classes
        for i in range(int(len(self.orig_classes) / 100)):
            if np.amax(self.orig_classes[i]) > 1:
                print("Converting classes to categorical")
                self.classes_categorical.value = to_categorical(np.asarray(self.orig_classes))
                break

        print("Original class amt", len(self.classes_categorical.value))

        if self.classes_freq_cutoff > 0:
            self.filtered_classes.value, self.filtered_class_names.value = proj.removeInfrequent(
                self.classes_categorical.value,
                self.orig_class_names,
                self.classes_freq_cutoff)
        else:
            self.filtered_classes.value = self.classes_categorical.value
            self.filtered_class_names.value = self.orig_class_names

        print("Final class amt", len(self.filtered_classes.value))
        super().process()

    def getClasses(self):
        return self.save_class.load(self.filtered_classes)

    def getClassNames(self):
        return self.save_class.load(self.filtered_class_names)

def getVocabStreamed(text_corpus_fn):

    dct = Dictionary()
    # Get X lines
    i = 0
    with open(text_corpus_fn) as infile:
        for line in infile:
            tokenized_doc = line.split()
            dct.add_documents([tokenized_doc])
            print(i)
            i += 1

    vocab = dct.token2id
    # id2token needs to be used before it can be obtained, so here we use it arbitrarily
    _ = dct[0]
    id2token = dct.id2token
    return vocab, dct, id2token

def doc2bowStreamed(text_corpus_fn, dct, bowmin):
    dct.filter_extremes(no_below=bowmin)  # Most occur in at least 2 documents
    bow = []
    i = 0
    with open(text_corpus_fn) as infile:
        for line in infile:
            tokenized_doc = line.split()
            bow.append(dct.doc2bow(tokenized_doc))
            print(i)
            i += 1
    bow = corpus2csc(bow)
    vocab = dct.token2id
    return dct, bow, vocab

def filterBowStreamed(text_corpus_fn, dct, no_below, no_above):
    dct.filter_extremes(no_below=no_below, no_above=no_above)
    f_bow = []
    i = 0
    with open(text_corpus_fn) as infile:
        for line in infile:
            tokenized_doc = line.split()
            f_bow.append(dct.doc2bow(tokenized_doc))
            print(i)
            i += 1
    filtered_bow = corpus2csc(f_bow)
    filtered_vocab = dct.token2id
    return filtered_bow, list(dct.token2id.keys()), filtered_vocab