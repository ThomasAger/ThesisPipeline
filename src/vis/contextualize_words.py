import numpy as np
from util.vis import contextualizeWords, mapWordsToContext
from util import io
import os
data_type = "sentiment"
orig =  "../..\data_paper\experimental results\chapter 5/"+data_type+"/"
if data_type == "sentiment":
    fn = "num_stw_num_stw_200_AWVEmp_ndcg_2000_10000_0_"
    csv = io.read_csv(orig+"csv/" + "sentiment_num_stw_num_stw_100_D2V_10000_0Streamed_ndcg.csv", dtype=str).index.values
else:
    fn = "num_stw_num_stw_200_MDS_ndcg_2000_5000_0_"
    csv = io.read_csv(orig+"csv/" + "num_stw_num_stw_200_MDS_5000_0_ndcg.csv", dtype=str).index.values

words = io.import1dArray(orig + "dir/"+ fn + "words.txt")
dirs = np.load(orig + "dir/" + fn + "dir.npy")
#
if len(words) != len(dirs):
    dirs = dirs.transpose()
ctx_word_array_fn = orig + fn + "ctx.txt"
if os.path.exists(ctx_word_array_fn) is False:
    ctx_word_array = contextualizeWords(words, dirs, words, dirs)
    np.save(ctx_word_array, ctx_word_array_fn)
else:
    ctx_word_array = np.load(ctx_word_array_fn)

new_ctx = mapWordsToContext(csv, ctx_word_array)

io.write1dArray(new_ctx, ctx_word_array_fn[:-4] + "ctx_csv.txt")