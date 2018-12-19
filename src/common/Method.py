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