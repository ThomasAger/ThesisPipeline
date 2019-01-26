from util import check
from common.SaveLoadPOPO import SaveLoadPOPO

# Can have many ways to define filenames
class Method:

    popo_array = None
    save_class = None
    file_name = None
    processed = False

    def __init__(self, file_name, save_class):
        self.save_class = save_class
        self.file_name = file_name
        self.makePopos()
        self.makePopoArray()

    def process_and_save(self):
        popo_array = self.popo_array
        if self.save_class.exists(popo_array) is False:
            print(self.__class__.__name__, "Doesn't exist, creating")
            self.process()
            self.makePopoArray()
            self.processed = True
            self.save_class.save(popo_array)
            print("corpus done")
        else:
            if self.save_class.load_all:
                self.save_class.loadAll(popo_array)
            print(self.__class__.__name__, "Already exists (lazy loading enabled)")

    def makePopos(self):
        print("Creating popos")

    def makePopoArray(self):
        print("Creating popo array")

    def process(self):
        print("Process complete")

    def save(self):
        if self.process:
            self.save_class.save(self.popo_array)
        else:
            print("Use process() first, or just use process_and_save()")

    def load(self):
        self.save_class.load(self.popo_array)

# Only saves one thing, its predictions, and always uses x_train, y_train, y_test, x_test splits.
class ModelMethod(Method):

    x_train = None
    y_train = None
    x_test = None
    y_test = None
    test_predictions = None
    test_proba = None
    probability = None


    def __init__(self, x_train, y_train, x_test, y_test, file_name, save_class, probability):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.probability = probability
        check.check_splits(self.x_train, self.y_train, self.x_test, self.y_test)
        super().__init__(file_name, save_class)

    def makePopos(self):
        self.test_predictions = SaveLoadPOPO(self.test_predictions, self.file_name + ".npy", "npy")
        if self.probability:
            self.test_proba = SaveLoadPOPO(self.test_proba, self.file_name + ".npy", "npy")

    def makePopoArray(self):
        if self.probability:
            self.popo_array = [self.test_predictions, self.test_proba]
        else:
            self.popo_array = [self.test_predictions]

    def getPred(self):
        return self.save_class.load(self.test_predictions)

    def getProba(self):
        return self.save_class.load(self.test_proba)

class RepMethod(Method):

    rep = None
    output_folder = None
    dim = None

    def __init__(self, file_name, output_folder, save_class, dim):
        self.output_folder = output_folder
        self.dim = dim
        super().__init__(file_name, save_class)

    def makePopos(self):
        self.rep = SaveLoadPOPO(self.rep, self.output_folder + self.file_name + ".npy", "npy")

    def makePopoArray(self):
        self.popo_array = [self.rep]

    def getRep(self):
        return self.save_class.load(self.rep)