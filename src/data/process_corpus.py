from common import Method
from common.SaveLoadPOPO import SaveLoadPOPO
from util.text_utils import naiveTokenizer, getVocab, doc2bow, filterBow, removeEmpty, preprocess, removeStopWords, \
    tokensToIds, split_all, cleanLargeCorpus, getVocabStreamed, doc2bowStreamed, filterBowStreamed

# Has batch processing (haven't figured out how to use it yet)
# Retains punctuation e.g. won't -> "won't"
# Has many additional options
# Seems hard to get vocab, tokens (maybe misunderstanding)
# Retains brackets, etc , e.g. ["(i", "think", "so)"]
# Can make vocab from your own corpus, but has different quirks




# Removes punctuation "won't be done, dude-man." = ["wont", "be", "done", "dudeman"]
# Lowercase and deaccenting "cYkět" = ["cyket"]
# Converting to ID's requires separate process using those vocabs. More time
# Finds phrases using gensim, e.g. "mayor" "of" "new" "york" -> "mayor" "of" "new_york"


# This causes OOM error. Need to rework


# For sentiment etc


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
    bowdict = None
    filtered_dict = None
    processed_corpus_txt = None

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
        if self.processed is False:
            self.split_corpus.value = self.save_class.load(self.split_corpus)
        return self.split_corpus.value
    def getClasses(self):
        if self.processed is False:
            self.classes.value = self.save_class.load(self.classes)
        return self.classes.value
    def getBow(self):
        if self.processed is False:
            self.bow.value = self.save_class.load(self.bow)
        return self.bow.value
    def getFilteredBow(self):
        if self.processed is False:
            self.filtered_bow.value = self.save_class.load(self.filtered_bow)
        return self.filtered_bow.value

    def getDct(self):
        if self.processed is False:
            self.dct.value = self.save_class.load(self.dct)
        return self.dct.value


    def getBowDct(self):
        if self.processed is False:
            self.bowdict.value = self.save_class.load(self.bowdict)
        return self.bowdict.value
    def getFilteredDct(self):
        if self.processed is False:
            self.filtered_dict.value = self.save_class.load(self.filtered_dict)
        return self.filtered_dict.value

    def getAllWords(self):

        if self.processed is False:
            self.all_words.value = self.save_class.load(self.all_words)
        return self.all_words.value

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
        self.filtered_dict = SaveLoadPOPO(self.dct, standard_fn + "metadata/" + file_name + "_filtered_dict.pkl", "gensim")
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

        self.processed_corpus_txt = SaveLoadPOPO(self.processed_corpus_txt,
                                             output_folder + "corpus/" + file_name + "_corpus_processed.txt", "1dtxt")

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
                           self.word_list, self.all_words,  self.split_corpus, self.bowdict, self.filtered_dict,
                           self.processed_corpus_txt]

    def process(self):
        print("Original doc len", len(self.orig_corpus))
        self.processed_corpus.value = preprocess(self.orig_corpus)
        self.tokenized_corpus.value = naiveTokenizer(self.processed_corpus.value)
        if self.remove_stop_words:
            self.tokenized_corpus.value, self.processed_corpus.value = removeStopWords(self.tokenized_corpus.value)
        self.processed_corpus.value, self.tokenized_corpus.value, self.remove_ind.value, self.classes.value = removeEmpty(self.processed_corpus.value,
                                                                                                                          self.tokenized_corpus.value, self.orig_classes)
        self.processed_corpus_txt.value = self.processed_corpus.value
        self.split_corpus.value = split_all(self.processed_corpus.value)

        self.all_vocab.value, self.dct.value, self.id2token.value = getVocab(self.tokenized_corpus.value)

        self.bowdict.value, self.bow.value, self.bow_vocab.value = doc2bow(self.tokenized_corpus.value, self.bowmin)

        self.all_words.value = list(self.bow_vocab.value.keys())
        print(self.bowmin, len(self.all_words.value), "|||", self.bow.value.shape)
        self.filtered_bow.value, self.word_list.value, self.filtered_vocab.value, self.filtered_dict.value = filterBow(self.tokenized_corpus.value,
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
        self.filtered_dict = SaveLoadPOPO(self.dct, standard_fn + "metadata/" + file_name + "_filtered_dict.pkl", "gensim")
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
                           self.word_list, self.all_words, self.bowdict, self.filtered_dict]
    def process(self):
        self.classes.value = self.orig_classes

        self.all_vocab.value, self.dct.value, self.id2token.value = getVocabStreamed(self.corpus_fn_to_stream)

        self.bowdict.value, self.bow.value, self.bow_vocab.value = doc2bowStreamed(self.corpus_fn_to_stream, self.bowmin)
        self.all_words.value = list(self.bowdict.value.keys())
        print(self.bowmin, len(self.all_words.value), "|||", self.bow.value.shape)
        self.filtered_bow.value, self.word_list.value, self.filtered_vocab.value, self.filtered_dict.value = filterBowStreamed(
            self.corpus_fn_to_stream,
            self.no_below, self.no_above)
        super().process()
        # We are not doing any changes to the classes/documents

# Does not support basic tokenization or clean up methods yet, only gensim methods
class LargeCorpus(MasterCorpus):
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
        self.filtered_dict = SaveLoadPOPO(self.dct, standard_fn + "metadata/" + file_name + "_filtered_dict.pkl", "gensim")
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
                           self.word_list, self.all_words, self.bowdict, self.filtered_dict]
    def process(self):
        self.classes.value = self.orig_classes

        cleaned_corpus_fn = self.output_folder + "corpus/" + self.file_name + "_corpus_processed.txt"
        cleanLargeCorpus(self.corpus_fn_to_stream, cleaned_corpus_fn)

        self.all_vocab.value, self.dct.value, self.id2token.value = getVocabStreamed(cleaned_corpus_fn)

        self.bowdict.value, self.bow.value, self.bow_vocab.value = doc2bowStreamed(self.corpus_fn_to_stream, self.bowmin)
        self.all_words.value = list(self.bowdict.value.keys())
        print(self.bowmin, len(self.all_words.value), "|||", self.bow.value.shape)
        self.filtered_bow.value, self.word_list.value, self.filtered_vocab.value, self.filtered_dict.value = filterBowStreamed(
            self.corpus_fn_to_stream,
            self.no_below, self.no_above)
        super().process()
        # We are not doing any changes to the classes/documents

