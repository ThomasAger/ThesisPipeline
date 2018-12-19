from util import check
from common.SaveLoadPOPO import SaveLoadPOPO

# Can have many ways to define filenames
class Method:

    popo_array = None
    save_class = None

    def __init__(self, save_class):
        self.save_class = save_class
        self.makePopos()
        self.makePopoArray()

    def process_and_save(self):
        popo_array = self.popo_array
        if self.save_class.exists(popo_array) is False:
            print(self.__class__.__name__, "Doesn't exist, creating")
            self.process()
            self.save_class.save(popo_array)
            print("corpus done")
        else:
            self.save_class.load(popo_array)
            print(self.__class__.__name__, "Already exists")

    def makePopos(self):
        print("Creating popos")

    def makePopoArray(self):
        print("Creating popo array")

    def process(self):
        self.makePopoArray()

    def save(self):
        self.save_class.save(self.popo_array)

    def load(self):
        self.save_class.load(self.popo_array)

# Only saves one thing, its predictions, and always uses x_train, y_train, y_test, x_test splits.
class ModelMethod(Method):

    x_train = None
    y_train = None
    x_test = None
    y_test = None
    test_predictions = None

    def __init__(self, x_train, y_train, x_test, y_test, save_class):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        check.check_x(x_train)
        check.check_x(x_test)
        check.check_y(y_train)
        check.check_y(y_test)
        check.check_splits(x_train, y_train, x_test, y_test)
        super().__init__(save_class)

    def makePopos(self):
        self.test_predictions = SaveLoadPOPO(self.test_predictions, self.file_name, "npy")

    def makePopoArray(self):
        self.popo_array = [self.test_predictions]