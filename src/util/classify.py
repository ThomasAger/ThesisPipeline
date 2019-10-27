import numpy as np
# From 2d array, change any values above 0 to 1
from common.SaveLoadPOPO import SaveLoadPOPO
from data.process_corpus import MasterCorpus
from util import py, proj
from keras.utils import to_categorical

def toBinary(x):
    for i in range(len(x)):
        for j in range(len(x[i])):
            if x[i][j] > 0:
                x[i][j] = 1
    return x

# Get the document frequencies for a corpus
def getDocumentFrequency(x):
    if len(x) < len(x[0]):
        raise ValueError("The array is reversed. (Amount of words < Amount of documents)")
    doc_freq = np.zeros(len(x))
    for i in range(len(x)): #  Words
        for j in range(len(x[i])): # Documents
            if x[i][j] > 0:
                doc_freq[x] += 1
    return doc_freq

def limitDocumentFrequency(doc_freq, word_by_doc, word_list, min_freq, max_freq):
    limit_freq_ind = np.where(doc_freq > min_freq)
    print("Removed terms below", min_freq, "doc frequency", len(limit_freq_ind), "terms remaining")
    word_list = word_list[limit_freq_ind]
    doc_freq = doc_freq[limit_freq_ind]
    word_by_doc = word_by_doc[limit_freq_ind]

    limit_freq_ind = np.where(doc_freq > (len(doc_freq) - max_freq))
    print("Removed terms above", (len(doc_freq) - max_freq), "doc frequency", len(limit_freq_ind), "terms remaining")
    word_list = word_list[limit_freq_ind]
    doc_freq = doc_freq[limit_freq_ind]
    word_by_doc = word_by_doc[limit_freq_ind]

    return word_list, word_by_doc, doc_freq


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