import numpy as np
from util import sim
import math
orig_fn = "../../data/processed/anime/"
pca_fn = orig_fn + "rep/pca/num_stw_50_PCA.npy"
pca = np.load(pca_fn)
names_fn = orig_fn + "corpus/entity_names_del.npy"
names = np.asarray(np.load(names_fn))

#0 = cowboy bebop
#115 = hunter x hunter
#096 = samurai champloo
#092 = spirited away
#1228 = sonic the hedgehog
#1-66 = gurrenn laggann
to_check = 101


for i in range(len(names)):
    print(i, names[i])

sims = []
for i in range(len(pca)):
    simp = sim.getSimilarity(pca[to_check], pca[i])
    if math.isnan(simp):
        print(i, "NAN", pca[i])
        sims.append(0.0)
    else:
        sims.append(simp)
top_ids = np.asarray(np.flipud(np.argsort(sims))[:20])
print(names[top_ids])
print(np.asarray(sims)[top_ids])