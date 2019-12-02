from util import vis as u
import numpy as np
from util import io
import pandas as pd
import os
data_type_1 = "newsgroups"
common_array_1 = "../../data_paper\experimental results\chapter 5/"+data_type_1+"/dir/common_all.npy"
fn_1 = "num_stw_num_stw_50_D2V_10000_0_"
fn_2 = "num_stw_US_5_Activ_tanh_Dropout_0.5_Hsize_[1000, 100]_mlnrep_10000_0_"
fn_3 = "num_stw_US_200_Activ_tanh_Dropout_0.1_Hsize_3_mlnrep_10000_0_"
data_type_2 = "movies"
common_array_2 = "../../data_paper\experimental results\chapter 5/"+data_type_2+"/dir/common_all.npy"
fn_4 = "num_stw_num_stw_200_MDS_10000_0_"
fn_5 = "num_stw_US_20_Activ_tanh_Dropout_0.5_Hsize_[1000, 100]_mlnrep_10000_0_"
fn_6 = "num_stw_US_300_Activ_tanh_Dropout_0.25_Hsize_3_mlnrep_10000_0_"
data_type_3 = "placetypes"
common_array_3 = "../../data_paper\experimental results\chapter 5/"+data_type_3+"/dir/common_all.npy"
fn_7 = "num_stw_num_stw_50_AWVEmp_10000_0_"
fn_8 = "num_stw_US_200_Activ_tanh_Dropout_0.5_Hsize_[1000, 100]_BS_10_mlnrep_10000_0_"
fn_9 = "num_stw_US_100_Activ_tanh_Dropout_0.25_Hsize_2_BS_10_mlnrep_10000_0_"

rewrite = False
common_arrays = [common_array_1, common_array_2, common_array_3]

fns = [[fn_1, fn_2, fn_3], [fn_4, fn_5, fn_6], [fn_7, fn_8, fn_9]]
data_types = [data_type_1,data_type_2,data_type_3]
# For each common csv
for i in range(len(common_arrays)):
    csv = np.load(common_arrays[i])
    # Get the three different contexts
    ctxs = []
    indexes = []
    for j in range(len(fns)):
        io.write1dArray(u.getPretty(u.mapWordsToContext(csv,
        np.load("../../data_paper\experimental results\chapter 5/"+data_types[i]+"/all_dir/"+fns[i][j]+"words_ctx.npy"))),
        common_arrays[i][:-4] + fns[i][j] + "_ctx.txt")
