import numpy as np
from util import io
data_types = ["placetypes", "sentiment", "newsgroups", "movies", "reuters"]
classes = [["OpenCYC","Foursquare","Geonames"], ["Sentiment"], ["Newsgroups"], ["Genres", "Keywords", "Ratings"], ["Reuters"]]

for i in range(len(data_types)):
    for j in range(len(classes[i])):
        path = "../../data/processed/" + data_types[i] + "/classes/num_stw"+classes[i][j]+"_fil_classes.npy"
        class_name_path = "../../data/processed/" + data_types[i] + "/classes/num_stw"+classes[i][j]+"_class_names.txt"
        class_all = np.load(path)
        if data_types[i] != "sentiment":
            if len(class_all) > len(class_all[0]):
                class_all = class_all.transpose()
            class_names = io.import1dArray(class_name_path)
            nonzeros = []
            print(data_types[i], classes[i][j])
            for k in range(len(class_names)):
                print(np.count_nonzero(class_all[k]))
                nonzeros.append(np.count_nonzero(class_all[k]))
            print(data_types[i], classes[i][j], int(round(np.average(nonzeros), 0)))
        else:
            print("Sentiment ")
            print(np.count_nonzero(class_all))
