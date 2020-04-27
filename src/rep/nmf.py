
from sklearn.decomposition import NMF as NMF_sk

from common.SaveLoadPOPO import SaveLoadPOPO
from common.Method import RepMethod
class NMF(RepMethod):
    word_doc_matrix = None
    doc_amt = None
    def __init__(self, matrix, doc_amt, dim, file_name, output_folder, save_class):

        self.matrix = matrix
        self.doc_amt = doc_amt
        super().__init__(file_name, output_folder, save_class, dim)

    def checkWordDocMatrix(self, doc_amt):
        if self.matrix.shape[1] != doc_amt and self.matrix.shape[0] != doc_amt and doc_amt > 0:
            raise ValueError("Incorrect number of documents", doc_amt,self.matrix.shape[1] )
        # Check if the words, typically the more frequent, are the rows or the columns, and transpose so they are the columns
        if self.matrix.shape[1] == doc_amt:
            self.matrix = self.matrix.transpose()


    def process(self):
        #self.checkWordDocMatrix(self.doc_amt)
        svd = NMF_sk(n_components=self.dim, verbose=True)  # use the scipy algorithm "arpack"
        self.rep.value = svd.fit_transform(self.matrix)
        super().process()
