import util.io as io




def renameIndex(score_csv):
    for i in range(len(score_csv.index.values)):
    
        replacement_string = ""
        if "MDS" in score_csv.index.values[i]:
            replacement_string += "MDS"
        elif "PCA" in score_csv.index.values[i]:
            replacement_string += "PCA"
        elif "AWV" in score_csv.index.values[i]:
            replacement_string += "AWV"
        elif "D2V" in score_csv.index.values[i]:
            replacement_string += "D2V"
    
        if "200"in score_csv.index.values[i].split("_")[4]:
            replacement_string += " 200"
        if "100"in score_csv.index.values[i].split("_")[4]:
            replacement_string += " 100"
        if "50" in score_csv.index.values[i].split("_")[4]:
            replacement_string += " 50"

        score_csv.index.values[i] = replacement_string
    return score_csv



def onlyTopRepType(score_csv):
    top_inds = []
    mds_array = []
    pca_array = []
    d2v_array = []
    awv_array = []
    for i in range(score_csv.shape[0]):
        if "MDS" in score_csv.index.values[i]:
            mds_array.append(i)
        if "PCA" in score_csv.index.values[i]:
            pca_array.append(i)
        if "AWV" in score_csv.index.values[i]:
            awv_array.append(i)
        if "D2V" in score_csv.index.values[i]:
            d2v_array.append(i)
            
    top_d2v = -1
    top_d2v_ind = -1
    for i in range(len(d2v_array)):
        val = score_csv.iloc[d2v_array[i],1]
        if val > top_d2v:
            top_d2v_ind = score_csv.index.values[d2v_array[i]]
            top_d2v = val

    top_pca = -1
    top_pca_ind = -1
    for i in range(len(pca_array)):
        val = score_csv.iloc[pca_array[i],1]
        if val > top_pca:
            top_pca_ind = score_csv.index.values[pca_array[i]]
            top_pca = val

    top_mds = -1
    top_mds_ind = -1
    for i in range(len(mds_array)):
        val = score_csv.iloc[mds_array[i],1]
        if val > top_mds:
            top_mds_ind = score_csv.index.values[mds_array[i]]
            top_mds = val

    top_awv = -1
    top_awv_ind = -1
    for i in range(len(awv_array)):
        val = score_csv.iloc[awv_array[i],1]
        if val > top_awv:
            top_awv_ind = score_csv.index.values[awv_array[i]]
            top_awv = val

    filtered_csv = None
    if top_mds_ind == -1:
        filtered_csv = score_csv.loc[[ top_pca_ind, top_awv_ind, top_d2v_ind ]]
    elif top_d2v_ind == -1:
        filtered_csv = score_csv.loc[[ top_pca_ind, top_awv_ind, top_mds_ind]]
    else:
        filtered_csv = score_csv.loc[[top_pca_ind, top_awv_ind,  top_mds_ind, top_d2v_ind ]]

    return filtered_csv

data_types = ["reuters", "sentiment", "placetypes", "newsgroups", "movies"]
score_types = ["rank", "rep", "clusters", "topic"]
for i in range(len(data_types)):
    for j in range(len(score_types)):
        file_name = "../../data/processed/" + data_types[i] + "/" + score_types[j] + "/score/csv_final/"
        fns = io.getFns(file_name)
        for k in range(len(fns)):
            if "replaced" not in fns[k] and "filtered" not in fns[k]:
                if fns[k][-4:] == ".csv":
                    df = io.read_csv(file_name + fns[k],error_bad_lines=False)
                    try:
                        renamed_csv = renameIndex(df)
                        renamed_csv.to_csv(file_name + fns[k][:-4]+"replaced.csv")
                        filtered_csv = onlyTopRepType(renamed_csv)
                        filtered_csv.to_csv(file_name + fns[k][:-4]+"filtered.csv")
                    except:
                        print("FAILED", file_name + fns[k])
