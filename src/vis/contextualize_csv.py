from util import vis as u
import numpy as np
from util import io
import pandas as pd
import os
data_type_1 = "newsgroups"
csv_1 = "../../data_paper\experimental results\chapter 5/"+data_type_1+"/csv/num_stw_num_stw_50_D2V_10000_0Streamed_ndcg.csv"
csv_2 = "../../data_paper\experimental results\chapter 5/"+data_type_1+"/csv/num_stw_US_5_Activ_tanh_Dropout_0.5_Hsize_[1000, 100]_mlnrep_10000_0Streamed_ndcg.csv"
csv_3 = "../../data_paper\experimental results\chapter 5/"+data_type_1+"/csv/num_stw_US_200_Activ_tanh_Dropout_0.1_Hsize_3_mlnrep_10000_0Streamed_ndcg.csv"
fn_1 = "num_stw_num_stw_50_D2V_10000_0_"
fn_2 = "num_stw_US_5_Activ_tanh_Dropout_0.5_Hsize_[1000, 100]_mlnrep_10000_0_"
fn_3 = "num_stw_US_200_Activ_tanh_Dropout_0.1_Hsize_3_mlnrep_10000_0_"
data_type_2 = "movies"
csv_4 = "../../data_paper\experimental results\chapter 5/"+data_type_2+"/csv/num_stw_num_stw_200_MDS_10000_0_ndcg.csv"
csv_5 = "../../data_paper\experimental results\chapter 5/"+data_type_2+"/csv/num_stw_US_20_Activ_tanh_Dropout_0.5_Hsize_[1000, 100]_mlnrep_10000_0_ndcg.csv"
csv_6 = "../../data_paper\experimental results\chapter 5/"+data_type_2+"/csv/num_stw_US_300_Activ_tanh_Dropout_0.25_Hsize_3_mlnrep_10000_0_ndcg.csv"
fn_4 = "num_stw_num_stw_200_MDS_10000_0_"
fn_5 = "num_stw_US_20_Activ_tanh_Dropout_0.5_Hsize_[1000, 100]_mlnrep_10000_0_"
fn_6 = "num_stw_US_300_Activ_tanh_Dropout_0.25_Hsize_3_mlnrep_10000_0_"
data_type_3 = "placetypes"
fn_7 = "num_stw_num_stw_50_AWVEmp_10000_0_"
fn_8 = "num_stw_US_200_Activ_tanh_Dropout_0.5_Hsize_[1000, 100]_BS_10_mlnrep_10000_0_"
fn_9 = "num_stw_US_100_Activ_tanh_Dropout_0.25_Hsize_2_BS_10_mlnrep_10000_0_"

csv_7 = "../../data_paper\experimental results\chapter 5/"+data_type_3+"/csv/num_stw_num_stw_50_AWVEmp_10000_0_F1_ACC_Kappa_.csv"
csv_8 = "../../data_paper\experimental results\chapter 5/"+data_type_3+"/csv/num_stw_US_200_Activ_tanh_Dropout_0.5_Hsize_[1000, 100]_BS_10_mlnrep_10000_0Streamed_ndcg.csv"
csv_9 = "../../data_paper\experimental results\chapter 5/"+data_type_3+"/csv/num_stw_US_100_Activ_tanh_Dropout_0.25_Hsize_2_BS_10_mlnrep_10000_0Streamed_ndcg.csv"
rewrite = False



csv_fns = [csv_7]
for i in range(len(csv_fns)):
    io.read_csv(csv_fns[i])

fns = [fn_7]
data_types = [data_type_3]
for i in range(len(csv_fns)):
    if True:#os.path.exists(csv_fns[i][:-4] + "_context.csv") is False and rewrite is False:
        print(csv_fns[i][:-4] + "_context.csv")
        csv = io.read_csv(csv_fns[i], dtype=str)
        ctx = np.load("../../data_paper\experimental results\chapter 5/"+data_types[i]+"/all_dir/"+fns[i]+"words_ctx.npy")
        new_index = u.getPretty(u.mapWordsToContext(csv.index.values, ctx))
        scores = csv.iloc[:,0].values
        new_data = pd.DataFrame(scores, index=new_index)
        new_data.to_csv(csv_fns[i][:-4] + "_context.csv")
