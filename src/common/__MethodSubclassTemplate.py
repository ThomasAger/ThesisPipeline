from common.SaveLoadPOPO import SaveLoadPOPO
from common.Method import Method
class MethodName(Method):
    popo_example = None
    param_example = None

    def __init__(self, param_example, save_class):

        self.param_example = param_example

        super().__init__(file_name, save_class)


    def makePopos(self):
        self.popo_example = SaveLoadPOPO(self.popo_example, "filename", "npy")

    def makePopoArray(self):
        self.popo_array = []

    def process(self):
        # Process
        super().process()