ranking = np.load("../../data/processed/movies/rank/fil/num_stw_num_stw_50_MDS_ndcg_1000_20000_0_rank.npy")
words = np.load("../../data/processed/movies/rank/fil/num_stw_num_stw_50_MDS_ndcg_1000_20000_0_words.npy")

entity_names = [2510,2399,2443,8762,446,2343,6384,3209,2435,429,6382]
ids = [1,74,9,0,20,13]

for i in range(len(entity_names)):
    thing = []
    for j in range(len(ids)):
        print(ranking[entity_names[i]][ids[j]], end =" ")
    print("")

ranking = ranking.transpose()

for j in range(len(ids)):
    print(np.amax(ranking[ids[j]]), end=" ")
    print(np.amin(ranking[ids[j]]), end=" ")
    print()
exit()
print("")
