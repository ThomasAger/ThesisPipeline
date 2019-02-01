import numpy as np
from util import sim
import math
orig_fn = "../../data/processed/anime/"
pca_fn = orig_fn + "rep/pca/num_stw_50_PCA.npy"
pca = np.load(pca_fn)
names_fn = orig_fn + "corpus/entity_names.npy"
names = np.asarray(np.load(names_fn))

#0 = cowboy bebop
#115 = hunter x hunter
#182 = samurai champloo

to_check = 0

for i in range(len(names)):
    print(i, names[i])

sims = []
for i in range(len(pca)):
    simp = sim.getSimilarity(pca[to_check], pca[i])
    if math.isnan(simp):
        print(pca[i])
        sims.append(0.0)
    else:
        sims.append(simp)
top_ids = np.asarray(np.flipud(np.argsort(sims))[:5])
print(names[top_ids])
print(np.asarray(sims)[top_ids])