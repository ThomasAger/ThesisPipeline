import numpy as np
words1 = np.load("../../data/processed/sentiment/rank/fil/num_stw_num_stw_100_D2V_ndcg_1000_10000_0_words.npy")
words2 = np.load("../../data/processed/sentiment/rank/fil/num_stw_num_stw_100_D2V_kappa_1000_10000_0_words.npy")
score1 = []
score2 = []

words_to_keep = []
for i in range(len(words2)):
    for j in range(len(words1)):
        if words2[i] == words1[j]:
            print("keep", words2[i])
            words_to_keep.append(words2[i])
            break
print(len(words_to_keep))
words_to_delete = []
for i in range(len(words2)):
    for j in range(len(words1)):
        if words2[i] == words1[j]:
            print("remove", words2[i])
            words_to_delete.append(i)
            break
print(len(words_to_delete))
words2 = np.delete(words2, words_to_delete, axis=0)
for i in range(len(words1)):
    for j in range(len(words2)):
        if words1[i] == words2[j]:
            print("remove", words1[i])
            words_to_delete.append(i)
            break
print(len(words_to_delete))
words1 = np.delete(words1, words_to_delete, axis=0)
print("----")
print(words1)
print("----")
print(words2)
print("----")
print(words1)
print("----")
print(words2)
print("----")
print(words_to_keep)
print("----")