import util.io as io
import util.py as py
import math
csv_name = "../../tex/tables/ranks_bold.csv"

csv = io.read_csv(csv_name,error_bad_lines=False)

"""
id_map_matrix = []
for i in range(len(latex)):
    id_map = []
    for j in range(len(latex[i])):
        try:
            float(latex[i][j])
            id_map.append(j)
        except:
            print(latex[i][j])

    id_map_matrix.append(id_map)
"""

for j in range(csv.shape[1]):
    for i in range(csv.shape[0]):
        try:
            if "Unnamed" in csv.values[i][j]:
                csv.values[i][j] = ""
            if str(csv.values[i][j]) == "nan":
                continue
            csv.iloc[i,j] = str(round(float(csv.values[i][j]), 3))
        except:
            print("")
ids_to_bold = []
for j in range(csv.shape[1]):
    ids = []
    top_float = 2147000000
    top_id = 0
    for i in range(csv.shape[0]):
        val = csv.values[i][j]
        if py.isStr(val) and "textbf" in val:
            continue
        try:
            if  float(val) < top_float:
                top_float = float(val)
                top_id = i
        except:
            print("except")
            if top_float < 2147000000:
                csv.iloc[top_id, j] = "\\underline{\\textit{" +  str(csv.values[top_id][j]) + "}}"
                ids.append(top_id)
            top_float = 2147000000
            top_id = 0
        if top_float < 2147000000 and i == csv.shape[0]-1:
            csv.iloc[top_id, j] = "\\underline{\\textit{" +  str(csv.values[top_id][j]) + "}}"
            ids.append(top_id)
    ids_to_bold.append(ids)

csv.to_csv(csv_name[:-4] + "_italic.csv")
print("")
"""
for j in range(csv.shape[1]):
    for i in range(csv.shape[0]):

new_latex = []
        print(i, j, id_map_matrix[i][ids_to_bold[i][j]])
        latex[i][id_map_matrix[i][ids_to_bold[i][j]]] =  latex[i][id_map_matrix[i][ids_to_bold[i][j]]] 
    new_latex.append(" ".join(latex[i]))

"""