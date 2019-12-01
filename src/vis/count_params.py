from util import io
from collections import Counter
import numpy as np

# Get all the best params for a domain/model_type

folder = "E:\PhD\Code\ThesisPipeline\ThesisPipeline\data\processed/placetypes/clusters\score\csv_averages/top_params/"

fns = io.getFns(folder)

best_params_all = []

for fn in fns:
    if ".npy" in fn:
        best_params = np.load(folder+fn)
    elif ".txt" in fn:
        best_params = io.import1dArray(folder + fn)
    for s in best_params:
        best_params_all.append(s)

# Count them
c = Counter(best_params_all)
print(c)