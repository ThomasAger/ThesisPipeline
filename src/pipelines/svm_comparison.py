import numpy as np
from model.svm import LogisticRegression, LinearSVM
import scipy.sparse as sp
from util.save_load import SaveLoad

from project.get_directions import GetDirections, GetDirectionsSimple
bow = sp.load_npz("E:\PhD\Code\ThesisPipeline\ThesisPipeline\data\processed/mafiascum/bow/num_stw_sparse_corpus.npz").toarray()
space = np.load("E:\PhD\Code\ThesisPipeline\ThesisPipeline\data\processed/mafiascum/rep\pca/num_stw_200_PCA.npy", allow_pickle=True)

bow = np.asarray(bow, dtype=np.int32)
bow[bow >= 1] = 1
bow[bow < 1] = 0
import time
start = time.time()
for i in range(len(bow)):
    LR_save = SaveLoad(rewrite=True)
    LR = LogisticRegression( space, bow[i], space, bow[i], "test", LR_save)
    LR.process_and_save()
    if i == 10:
        break
end = time.time()

print(end - start)

start = time.time()
for i in range(len(bow)):
    LR_save = SaveLoad(rewrite=True, no_save=True, verbose=False)
    LR = LogisticRegression( space, bow[i], space, bow[i], "test", LR_save, fast=True)
    LR.process_and_save()
    if i == 10:
        break

end = time.time()

print(end - start)

start = time.time()
for i in range(len(bow)):
    LR_save = SaveLoad(rewrite=True, no_save=True, verbose=False)
    LR = LinearSVM( space, bow[i], space, bow[i], "test", LR_save)
    LR.process_and_save()
    print(i)
    if i == 10:
        break
end = time.time()
print(end - start)

"""
dir_save = SaveLoad(rewrite=False)
dir = GetDirectionsSimple(bow, space, dir_save, "test" , "")
dir.process_and_save()
# Get rankings on directions save all of them in a word:ranking on entities format, and retrieve if already saved
dirs = dir.getDirections()
"""
