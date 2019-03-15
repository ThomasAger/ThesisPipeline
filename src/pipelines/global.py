import numpy as np
import util.io as dt
if __name__ == '__main__':
    rank = dt.loadNpyDict("../../data/processed/newsgroups/rank/ndcg/num_stw_num_stw_100_AWVEmp_ndcg_dir.npy")
    rank_dir = dt.loadNpyDict("../../data/processed/newsgroups/rank/num_stw_num_stw_100_AWVEmp_rank_dir.npy")
    ranks_50 = np.load("../../data/processed/newsgroups/rank/num_stw_num_stw_100_AWVEmp_50_0_rank.npy")
    awv = np.load("D:\PhD\Code\ThesisPipeline\ThesisPipeline\data\processed/newsgroups/awv/num_stw_100_AWVEmp.npy")
    dir = np.load("../../data/processed/newsgroups/directions/dir/num_stw_num_stw_100_AWVEmp_50_0_dir.npy")
    dict_dir = dt.loadNpyDict("../../data/processed/newsgroups/directions/num_stw_num_stw_100_AWVEmp_word_dir.npy")
    print("k")