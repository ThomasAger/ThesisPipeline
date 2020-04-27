from gensim.corpora import Dictionary
from gensim.utils import deaccent
from gensim.models.phrases import Phraser
from nltk.corpus import stopwords
import numpy as np
import string
from sklearn.feature_extraction.text import CountVectorizer
from gensim.matutils import corpus2csc
from common import Method
import re
from common.SaveLoadPOPO import SaveLoadPOPO


# PAT_ALPHABETIC = re.compile(r'(((?![\d])\w)+)', re.UNICODE) # stolen from gensim
PAT_ALPHANUMERIC = re.compile(r'((\w)+)', re.UNICODE)  # stolen from gensim, but added numbers included

def tokenize(text):
    for match in PAT_ALPHANUMERIC.finditer(text):
        yield match.group()


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


def doc2bow(tokenized_corpus, bowmin):
    dct = Dictionary(tokenized_corpus)
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


def filterBow(tokenized_corpus, no_below, no_above):
    dct = Dictionary(tokenized_corpus)
    dct.filter_extremes(no_below=no_below, no_above=no_above)
    filtered_bow = [dct.doc2bow(text) for text in tokenized_corpus]
    filtered_bow = corpus2csc(filtered_bow)
    filtered_vocab = dct.token2id
    return filtered_bow, list(dct.token2id.keys()), filtered_vocab, dct


def removeEmpty(processed_corpus, tokenized_corpus, classes):
    remove_ind = []
    for i in range(len(processed_corpus)):
        if len(tokenized_corpus[i]) <= 0:
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


def ngrams(tokenized_corpus):  # Increase the gram amount by 1
    processed_corpus = np.empty(len(tokenized_corpus), dtype=np.object)
    phrases = Phrases(tokenized_corpus)
    gram = Phraser(phrases)
    for i in range(len(tokenized_corpus)):
        tokenized_corpus[i] = gram[tokenized_corpus[i]]
        processed_corpus[i] = " ".join(tokenized_corpus[i])
    return processed_corpus, tokenized_corpus



def makeDictStreamed(text_corpus_fn):
    dct = Dictionary()
    # Get X lines
    i = 0
    with open(text_corpus_fn) as infile:
        for line in infile:
            tokenized_doc = line.split()
            dct.add_documents([tokenized_doc])
            print(i)
            i += 1
    return dct


def getVocabStreamed(text_corpus_fn):
    dct = makeDictStreamed(text_corpus_fn)
    vocab = dct.token2id
    # id2token needs to be used before it can be obtained, so here we use it arbitrarily
    _ = dct[0]
    id2token = dct.id2token
    return vocab, dct, id2token


def doc2bowStreamed(text_corpus_fn, bowmin):
    dct = makeDictStreamed(text_corpus_fn)
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


def filterBowStreamed(text_corpus_fn,  no_below, no_above):
    dct = makeDictStreamed(text_corpus_fn)
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
    return filtered_bow, list(dct.token2id.keys()), filtered_vocab, dct

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


def cleanLargeCorpus(corpus_fn_to_stream, corpus_fn_to_save):
    space_table = str.maketrans(dict.fromkeys("\n\r", " "))
    punc_table = str.maketrans(dict.fromkeys(string.punctuation))
    stop_words = set(stopwords.words('english'))
    c = 0
    with open(corpus_fn_to_save, 'a') as write_file:
        with open(corpus_fn_to_stream) as infile:
            for line in infile:
                processed_line = line.translate(space_table)
                # Lowercase
                processed_line = processed_line.lower()
                # Remove all punctuation
                processed_line = processed_line.translate(punc_table)
                # Replace all whitespace with single whitespace
                processed_line = re.sub(r'\s+', ' ', processed_line)
                # Deaccent
                processed_line = deaccent(processed_line)
                # Strip trailing whitespace
                processed_line = processed_line.strip()
                processed_line = list(tokenize(processed_line))
                for j in reversed(range(len(processed_line))):
                    if len(processed_line[j]) == 1:
                        del processed_line[j]

                processed_line = [w for w in processed_line if w not in stop_words]
                processed_line = " ".join(processed_line)

                if len(processed_line) > 0:
                    c += 1
                    write_file.write(processed_line + "\n")
                    print(processed_line)
    print("Len of docs", c)
class LimitWordsMaster(Method.Method):

    word_list = None
    output_folder = None
    bow_word_dict = None
    dct = None
    bow = None

    def __init__(self, file_name, save_class, dct, bow, output_folder, word_list):
        self.word_list = word_list
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



    def makePopoArray(self):
        self.popo_array = [self.bow_word_dict, self.new_word_dict]



class LimitWords(LimitWordsMaster):
    no_below = None
    no_above = None

    def __init__(self, file_name, save_class, dct, bow, output_folder, word_list, no_below, no_above):
        self.no_above = no_above
        self.no_below = no_below
        super().__init__(file_name, save_class, dct, bow, output_folder, word_list)

    def process(self):
        orig_dct = self.dct.token2id
        self.dct.filter_extremes(no_below=self.no_below, no_above=self.no_above)
        self.bow_word_dict.value = dict(self.dct.token2id)
        self.new_word_dict.value = dict(self.dct.token2id)
        for key, value in self.bow_word_dict.value.items():
            self.bow_word_dict.value[key] = orig_dct[key]
        super().process()

    def makePopos(self):
        self.bow_word_dict = SaveLoadPOPO(None, self.output_folder + self.file_name + "_wldct_NB_" + str(self.no_below) + "_NA_" + str(self.no_above) + ".npy", "npy_dict")
        self.new_word_dict = SaveLoadPOPO(None, self.output_folder + self.file_name + "_new_wdct_NB_" + str(self.no_below) + "_NA_" + str(self.no_above) + ".npy", "npy_dict")


class LimitWordsNumeric(LimitWordsMaster):
    top_freq = None

    def __init__(self, file_name, save_class, dct, bow, output_folder, word_list, top_freq):
        self.top_freq = top_freq
        super().__init__(file_name, save_class, dct, bow, output_folder, word_list)

    def makePopos(self):
        self.bow_word_dict = SaveLoadPOPO(None, self.output_folder + self.file_name + "_wldct_NB_" + str(self.top_freq) + "_NA_" + str(0) + ".npy", "npy_dict")
        self.new_word_dict = SaveLoadPOPO(None, self.output_folder + self.file_name + "_new_wdct_NB_" + str(self.top_freq) + "_NA_" + str(0) + ".npy", "npy_dict")

    def process(self):
        ids = np.asarray(list(self.dct.dfs.keys()))
        freqs = np.asarray(list(self.dct.dfs.values()))
        s_freqs = np.flipud(np.argsort(freqs))[:self.top_freq]
        self.bow_word_dict.value = {}
        self.new_word_dict.value = {}
        __unused = self.dct[0]
        ids = ids[s_freqs]
        for key in ids:
            self.bow_word_dict.value[self.dct.id2token[key]] = self.dct.token2id[self.dct.id2token[key]]
        for i in range(len(list(self.bow_word_dict.value.keys()))):
            self.new_word_dict.value[list(self.bow_word_dict.value.keys())[i]] = i
        super().process()

