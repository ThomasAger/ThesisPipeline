import util.io as io
import util.py as py
import math
csv_name = "../../tex/tables/big results table.csv"
latex_name = "../../tex/tables/latex/all_rep_latex.txt"

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
    top_float = 0.0
    top_id = 0
    for i in range(csv.shape[0]):
        val = csv.values[i][j]

        try:
            if  float(val) > top_float:
                top_float = float(val)
                top_id = i
        except:
            print("except")
            if top_id > 0.0:
                csv.iloc[top_id, j] = "\\textbf{" +  str(csv.values[top_id][j]) + "}"
                ids.append(top_id)
            top_float = 0.0
            top_id = 0
        if top_float > 0.0 and i == csv.shape[0]-1:
            csv.iloc[top_id, j] = "\\textbf{" +  str(csv.values[top_id][j]) + "}"
            ids.append(top_id)
    ids_to_bold.append(ids)

csv.to_csv(csv_name[:-4] + "_bold.csv")
print("")
"""
for j in range(csv.shape[1]):
    for i in range(csv.shape[0]):

new_latex = []
        print(i, j, id_map_matrix[i][ids_to_bold[i][j]])
        latex[i][id_map_matrix[i][ids_to_bold[i][j]]] =  latex[i][id_map_matrix[i][ids_to_bold[i][j]]] 
    new_latex.append(" ".join(latex[i]))

"""