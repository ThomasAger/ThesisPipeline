from common.SaveLoadPOPO import SaveLoadPOPO
from common.Method import Method
from sklearn import preprocessing
class NormalizeZeroMean(Method):
    popo_example = None
    output_folder = None
    multidim_array = None
    normalized_output = None

    def __init__(self, multidim_array, file_name, output_folder, save_class):

        self.output_folder = output_folder
        self.multidim_array = multidim_array

        super().__init__(file_name, save_class)

    def makePopos(self):
        self.normalized_output = SaveLoadPOPO(self.normalized_output, self.output_folder + self.file_name + ".npy", "npy")

    def makePopoArray(self):
        self.popo_array = [self.normalized_output]

    def process(self):
        # Process
        self.normalized_output.value = preprocessing.scale(self.multidim_array)
        print("completed")
        super().process()

    def getNormalized(self):
        if self.processed is False:
            self.normalized_output.value = self.save_class.load(self.normalized_output)
        return self.normalized_output.value