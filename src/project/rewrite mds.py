from util import nlp
#opencyc = np.load("D:\PhD\Code\ThesisPipeline\ThesisPipeline\data_request\Lucas email 1\data\classes/num_stwOpenCYC_classes.npy")

mds = dt.import2dArray("../../data/processed/movies/rep/mds/films200.txt")

print("end all names")

fil_names = np.load("../../data/raw/movies/fil_names_punct_removed.npy")

all_names = np.load("../../data/raw/movies/all_names_punct_removed.npy")
ids = []
for j in range(len(fil_names)):
    for i in range(len(all_names)):
        if fil_names[j] == all_names[i]:
            ids.append(i)
            break
    print(i)
print("End fill names")
print(all_names[ids])
dt.write1dArray(all_names[ids], "../../data/raw/movies/new_film_names.txt")

all_names = dt.import1dArray("../../data/raw/movies/new_film_names.txt")
for i in range(len(fil_names)):
    if fil_names[i] != all_names[i]:
        print(i)
        raise ValueError("Wrong")
np.save("../../data/raw/movies/id_mapping.txt", ids)

mds = np.asarray(mds)
mds = mds[ids]
np.save("../../data/processed/movies/rep/mds/num_stw_200_MDS.npy", mds)
if len(mds) != 13978:
    raise ValueError("?????")
