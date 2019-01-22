from common.Method import Method
from common.SaveLoadPOPO import SaveLoadPOPO
from util import py
from util import io as dt
from util.save_load import SaveLoad
import numpy as np

class GetMetadata(Method):

    classes = None
    class_names = None
    output_fn = None
    class_metadata = None

    def __init__(self, classes, class_names, file_name, output_fn, save_class):

        self.classes = classes
        self.class_names = class_names
        self.output_fn = output_fn

        self.classes = py.transIfRowsLarger(self.classes)

        super().__init__(file_name, save_class)


    def makePopos(self):
        self.class_metadata = SaveLoadPOPO(self.class_metadata, self.output_fn + self.file_name + ".csv", "csv")

    def makePopoArray(self):
        self.popo_array = [self.class_metadata]

    def process(self):
        nonzero_count = []
        zero_count = []
        len_count = []
        props = []
        for i in range(len(self.classes)):
            nonzero_count.append(np.count_nonzero(self.classes[i]))
            len_count.append(len(self.classes[i]))
            zero_count.append(len_count[i] - nonzero_count[i])
            props.append(zero_count[i] / nonzero_count[i])

        self.class_metadata.value = [["len", "zeros", "ones", "prop"], [len_count, zero_count, nonzero_count, props], self.class_names]

        super().process()

if __name__ == '__main__':
    save_class = SaveLoad()
    data_type = "reuters"
    output_fn = "../../data_paper/classes/"
    file_name = "nonzero_count_" + data_type
    data_output_fn = "../../data/processed/" + data_type + "/classes/"
    classes = np.load(data_output_fn + "num_classes.npy")
    class_names = dt.import1dArray(data_output_fn + "num_stw_class_names.txt")
    md = GetMetadata(classes, class_names, file_name, output_fn, save_class)
    md.process_and_save()
