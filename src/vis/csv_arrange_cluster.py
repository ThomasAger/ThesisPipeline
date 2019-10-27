import util.io as io


def renameIndex(score_csv):
    for i in range(len(score_csv.index.values)):

        replacement_string = ""
        if "derrac" in score_csv.index.values[i]:
            replacement_string += "Derrac"
            if "200" in score_csv.index.values[i].split("_")[12]:
                replacement_string += " 200"
            if "100" in score_csv.index.values[i].split("_")[12]:
                replacement_string += " 100"
            if "50" in score_csv.index.values[i].split("_")[12]:
                replacement_string += " 50"
        elif "k-means" in score_csv.index.values[i]:
            replacement_string += "K-Means"
            if "200" in score_csv.index.values[i].split("_")[15]:
                replacement_string += " 200"
            if "100" in score_csv.index.values[i].split("_")[15]:
                replacement_string += " 100"
            if "50" in score_csv.index.values[i].split("_")[15]:
                replacement_string += " 50"

        score_csv.index.values[i] = replacement_string
    return score_csv


def rearrange_csv(score_csv):
    new_csv_values = [None, None, None, None, None, None]
    new_csv_index = [None, None, None, None, None, None]
    for i in range(len(score_csv.index.values)):
        if "K-Means" in score_csv.index.values[i] and "200" in score_csv.index.values[i]:
            new_csv_values[0] = score_csv.values[i]
            new_csv_index[0] = score_csv.index.values[i]
        if "K-Means" in score_csv.index.values[i] and "100" in score_csv.index.values[i]:
            new_csv_values[1] = score_csv.values[i]
            new_csv_index[1] = score_csv.index.values[i]
        if "K-Means" in score_csv.index.values[i] and "50" in score_csv.index.values[i]:
            new_csv_values[2] = score_csv.values[i]
            new_csv_index[2] = score_csv.index.values[i]
        if "Derrac" in score_csv.index.values[i] and "200" in score_csv.index.values[i]:
            new_csv_values[3] = score_csv.values[i]
            new_csv_index[3] = score_csv.index.values[i]
        if "errac" in score_csv.index.values[i] and "100" in score_csv.index.values[i]:
            new_csv_values[4] = score_csv.values[i]
            new_csv_index[4] = score_csv.index.values[i]
        if "errac" in score_csv.index.values[i] and "50" in score_csv.index.values[i]:
            new_csv_values[5] = score_csv.values[i]
            new_csv_index[5] = score_csv.index.values[i]

    for i in range(len(score_csv.values)):
        score_csv.iloc[i] = new_csv_values[i]
        score_csv.index.values[i] = new_csv_index[i]
    return score_csv


data_types = ["reuters", "sentiment", "placetypes", "newsgroups", "movies"]
score_types = [ "clusters"]
for i in range(len(data_types)):
    for j in range(len(score_types)):
        file_name = "../../data/processed/" + data_types[i] + "/" + score_types[j] + "/score/csv_final/"
        fns = io.getFns(file_name)
        for k in range(len(fns)):
            if "rearranged" not in fns[k] and "filtered" not in fns[k] and "replaced" not in fns[k] and "True" not in fns[k]:
                if fns[k][-4:] == ".csv":
                    df = io.read_csv(file_name + fns[k], error_bad_lines=False)
                    try:
                        renamed_csv = renameIndex(df)
                        filtered_csv = rearrange_csv(renamed_csv)
                        filtered_csv.to_csv(file_name + fns[k][:-4] + "rearranged.csv")
                    except:
                        print("fail")

