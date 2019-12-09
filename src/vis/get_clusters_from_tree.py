import numpy as np
from util import io
import pydotplus
from util import py
from util import vis
data_type = "movies"
folder1, folder_fn1 = io.getFns("../../data_paper\experimental results\chapter 5/"+data_type+"/trees/unsupervised")
folder2, folder_fn2 = io.getFns("../../data_paper\experimental results\chapter 5/"+data_type+"/trees/bow")
folder3, folder_fn3 = io.getFns("../../data_paper\experimental results\chapter 5/"+data_type+"/trees/vector")
save_fn1 = "../../data_paper\experimental results\chapter 5/"+data_type+"/tree_labels/"
split_threshold1 = 100
data_type = "placetypes"
folder4, folder_fn4 = io.getFns("../../data_paper\experimental results\chapter 5/"+data_type+"/trees/unsupervised")
folder5, folder_fn5 = io.getFns("../../data_paper\experimental results\chapter 5/"+data_type+"/trees/bow")
folder6, folder_fn6 = io.getFns("../../data_paper\experimental results\chapter 5/"+data_type+"/trees/vector")
save_fn2 = "../../data_paper\experimental results\chapter 5/"+data_type+"/tree_labels/"
split_threshold2 = 100
data_type = "newsgroups"
folder7, folder_fn7 = io.getFns("../../data_paper\experimental results\chapter 5/"+data_type+"/trees/unsupervised")
folder8, folder_fn8 = io.getFns("../../data_paper\experimental results\chapter 5/"+data_type+"/trees/bow")
folder9, folder_fn9 = io.getFns("../../data_paper\experimental results\chapter 5/"+data_type+"/trees/vector")
save_fn3 = "../../data_paper\experimental results\chapter 5/"+data_type+"/tree_labels/"

add_folder = ["unsupervised", "bow", "vector","unsupervised", "bow", "vector","unsupervised", "bow", "vector"]
split_threshold3 = 100
folders = [folder1, folder2, folder3, folder4, folder5, folder6, folder7, folder8, folder9]
folder_fns = [folder_fn1, folder_fn2,folder_fn3,folder_fn4,folder_fn5,folder_fn6,folder_fn7,folder_fn8,folder_fn9]
splits = [split_threshold1,split_threshold1,split_threshold1,split_threshold2,split_threshold2,split_threshold2,split_threshold3,split_threshold3,split_threshold3]
save_fns = [save_fn1,save_fn1,save_fn1,save_fn2,save_fn2,save_fn2,save_fn3,save_fn3,save_fn3]
for z in range(len(folders)):
    classes = []
    dot_files = []
    for j in range(len(folders[z])):
        if ".dot" in folders[z][j]:
            dot_files.append(folder_fns[z] + "/"+ folders[z][j])
            classes.append(folders[z][j].split("3")[-1].split(".")[0])
    graphs = []
    for j in range(len(dot_files)):
        graphs.append(pydotplus.graphviz.graph_from_dot_file(dot_files[j]))
    node_label_graphs = []
    for j in range(len(graphs)):
        node_labels = []
        for key, value in graphs[j].obj_dict["nodes"].items():
            if py.isInt(py.convertToInt(key)):
                node_labels.append(value[0]["attributes"]["label"])
        node_label_graphs.append(node_labels)
    split_node_label_graphs = []
    for i in range(len(node_label_graphs)):
        split_node_labels = []
        for j in range(len(node_label_graphs[i])):
            split_node_labels.append(node_label_graphs[i][j].split("\\n"))
        split_node_label_graphs.append(split_node_labels)
    graph_dicts = []
    for i in range(len(split_node_label_graphs)):
        node_dict = []
        for j in range(len(split_node_label_graphs[i])):
            node = {}
            for k in range(len(split_node_label_graphs[i][j])):
                split_dict = split_node_label_graphs[i][j][k].split("=")
                split_dict[0] = split_dict[0].strip()
                if "node" not in split_dict[0]:
                    if "gini" != split_dict[0] and "samples" != split_dict[0] and "value" != split_dict[0] and "class" != split_dict[0]:
                        node["label"] = split_dict[0][:-2]
                    else:
                        node[split_dict[0].strip()] = split_dict[1]
            try:
                node["label"]
                node_dict.append(node)
            except:
                pass
            #print(node)
        graph_dicts.append(node_dict)

    class_labels = []
    double_class_labels = []
    class_trimmed_labels = []
    double_class_trimmed_labels = []
    for i in range(len(graph_dicts)):
        labels = []
        trimmed_labels = []
        for j in range(len(graph_dicts[i])):
            labels.append(graph_dicts[i][j]["label"])
            if float(graph_dicts[i][j]["samples"].strip("%")) >= splits[z]:
                trimmed_labels.append(graph_dicts[i][j]["label"])

        u_labels, counts = np.unique(labels, return_counts = True)
        u_t_labels, t_counts = np.unique(trimmed_labels, return_counts=True)
        double_labels = []
        double_t_labels = []
        for k in range(len(counts)):
            if counts[k] > 1:
                double_labels.append(labels[k])
        for k in range(len(t_counts)):
            if t_counts[k] > 1:
                double_t_labels.append(trimmed_labels[k])
        class_labels.append(u_labels)
        class_trimmed_labels.append(u_t_labels)
        double_class_labels.append(np.unique(double_labels))
        double_class_trimmed_labels.append(np.unique(double_t_labels))

    print(double_class_labels)
    np.save(save_fns[z] + add_folder[z] +"/" + "classes"+ ".npy", classes)
    np.save(save_fns[z] +  add_folder[z] +"/" + "all_tree_labels"+ ".npy", class_labels)
    np.save(save_fns[z] +  add_folder[z] +"/" + "double_tree_labels"+ ".npy", double_class_labels)
    np.save(save_fns[z] +  add_folder[z] +"/" + "all_tree_labels_trimmed_" + str(splits[z]) + ".npy", class_trimmed_labels)
    np.save(save_fns[z] +  add_folder[z] +"/" + "double_tree_labels_trimmed_" + str(splits[z]) + ".npy", double_class_trimmed_labels)



    c_l_p = vis.getPrettyStrings(class_labels)
    d_c_l = vis.getPrettyStrings(double_class_labels)
    c_t_l = vis.getPrettyStrings(class_trimmed_labels)
    d_c_t_l = vis.getPretty(double_class_trimmed_labels)

    io.write_csv(save_fns[z] + add_folder[z] +"/" + "all" + str(splits[z]) + ".csv", ["all_tree", "double_tree", "all_tree_trimmed", "double_tree_trimmed"],
                 [class_labels, double_class_labels, class_trimmed_labels, double_class_trimmed_labels], key=classes)

    io.write_csv(save_fns[z] +add_folder[z] +"/" +  "all_pretty" + str(splits[z]) + ".csv",
                 ["all_tree", "double_tree", "all_tree_trimmed", "double_tree_trimmed"],
                 [c_l_p, d_c_l, c_t_l, d_c_t_l], key=classes)

    print("kay")

# Get all dot files

# Get class

# For each class
# Get cluster: samples/gini arrays

# Cut-off based on those scores/samples/gini, etc

# Save as csv arranged in class/cluster format
