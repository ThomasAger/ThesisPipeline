
from sklearn.decomposition import TruncatedSVD

from common.SaveLoadPOPO import SaveLoadPOPO
from common.Method import Method
class PCA(Method):
    rep = None
    word_doc_matrix = None
    dim = None
    file_name = None

    def __init__(self, word_doc_matrix, doc_amt, dim, file_name, save_class):

        self.word_doc_matrix = word_doc_matrix
        self.checkWordDocMatrix(doc_amt)
        self.dim = dim
        self.file_name = file_name

        super().__init__(file_name, save_class)

    def checkWordDocMatrix(self, doc_amt):
        if self.word_doc_matrix.shape[1] != doc_amt and self.word_doc_matrix.shape[0] != doc_amt:
            raise ValueError("Incorrect number of documents")
        # Check if the words, typically the more frequent, are the rows or the columns, and transpose so they are the columns
        if self.word_doc_matrix.shape[1] == doc_amt:
            self.word_doc_matrix = self.word_doc_matrix.transpose()

    def makePopos(self):
        self.rep = SaveLoadPOPO(self.rep, self.file_name + "PCA.npy", "npy")

    def makePopoArray(self):
        self.popo_array = [self.rep]

    def process(self):
        print("Begin processing")
        svd = TruncatedSVD(n_components=self.dim)  # use the scipy algorithm "arpack"
        self.rep.value = svd.fit_transform(self.word_doc_matrix)
        super().process()
