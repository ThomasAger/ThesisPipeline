
from sklearn.decomposition import TruncatedSVD

from common.SaveLoadPOPO import SaveLoadPOPO
from common.Method import RepMethod
class PCA(RepMethod):
    word_doc_matrix = None

    def __init__(self, word_doc_matrix, doc_amt, dim, file_name, output_folder, save_class):

        self.word_doc_matrix = word_doc_matrix
        self.checkWordDocMatrix(doc_amt)

        super().__init__(file_name, output_folder, save_class, dim)

    def checkWordDocMatrix(self, doc_amt):
        if self.word_doc_matrix.shape[1] != doc_amt and self.word_doc_matrix.shape[0] != doc_amt:
            raise ValueError("Incorrect number of documents")
        # Check if the words, typically the more frequent, are the rows or the columns, and transpose so they are the columns
        if self.word_doc_matrix.shape[1] == doc_amt:
            self.word_doc_matrix = self.word_doc_matrix.transpose()

    def process(self):
        svd = TruncatedSVD(n_components=self.dim)  # use the scipy algorithm "arpack"
        self.rep.value = svd.fit_transform(self.word_doc_matrix)
        super().process()
