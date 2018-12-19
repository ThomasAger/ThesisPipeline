import os
import pandas as pd
from sklearn.externals import joblib
import numpy as np


def load_manage_csv(method_name):
    # If rewrite column doesnt exist, add it
    return pd.DataFrame()

def save_in_csv(method_name, param_dict, output_files, output_names):
    method_csv = load_manage_csv(method_name)
    # If any of the param names don't exist then add them as a column
    param_names = list(param_dict.keys())
    for i in range(len(param_names)):
        if param_names[i] not in method_csv.columns:
            method_csv[param_names[i]] = np.nan
    # If any of the output names don't exist then add them as a column
    for i in range(len(output_names)):
        if output_names[i] not in method_csv.columns:
            method_csv[output_names[i]] = np.nan
    output_dict = {}
    # Generate filenames for the outputs and try saving them in a file structure
    for i in range(len(output_files)):
        filename = generate_filename(method_name, param_names, output_names)
        save_file(output_files[i], filename)
        output_dict[output_names[i]] = filename
    
    # Add a new row with the parameters and full generated filenames for the output
    

    # Save the CSV
    """
    for i in range(len(output_files)): # Or whatever
        if filetypes[i] == dot_file: #string dot file to  create tree
            output_files[i].write_png(filename_manager.get_file_name())
        elif filetypes[i] == tree_file: #scikit-learn tree file, clf
            joblib.dump(output_files[i], model_name_fn)
    """
    return output_fns


def load_save(class_name, method, method_name, param_dict, output_names):
    if exists(method, method_name, param_dict, output_names):
        output_files, output_fns = load_files(method, method_name, param_dict, output_names)
    else:
        output_files = method(*param_dict)
        if len(output_files) != len(output_names):
            raise ValueError("Amount of output names does not match amount of files")
        output_fns = save_in_csv(method_name, param_dict, output_files, output_names)
    output_dict = {}
    for i in range(len(output_names)):
        output_dict[output_names[i]] = {"file":output_files[i], "fn":output_fns[i]}
    # Where output_files is a dict of param_name:file, and output_fns is a dict of param_name:fn
    return output_dict

def getCSVsToAverage(csv_folder_fn,  starting_fn=""):
    fns = getFns(csv_folder_fn)
    fns_to_average = []
    try:
        cross_val = int(starting_fn.split()[0][len(starting_fn.split()[1]) - 3])
    except ValueError:
        try:
            cross_val = int(starting_fn.split()[1][len(starting_fn.split()[1]) - 3])
        except ValueError:
            cross_val = 12354432
        except IndexError:
            cross_val = 123131355
    except IndexError:
        cross_val = 9898989

    og_st_fn, st_fn = removeCSVText(starting_fn)
    print(og_st_fn)
    for f in fns:
        if len(st_fn) > 0:
            og_f, cut_fn = removeCSVText(f)
            try:
                cross_val_cut_fn = int(f.split()[0][len(f.split()[0])-3])
            except ValueError:
                try:
                    cross_val_cut_fn = int(f.split()[1][len(f.split()[1]) - 3])
                except ValueError:
                    cross_val_cut_fn = 1235334432
                except IndexError:
                    cross_val_cut_fn = 12333131355
            except IndexError:
                cross_val_cut_fn = 232322
            if st_fn == cut_fn and cross_val == cross_val_cut_fn:
                print(og_f)
                print(cut_fn)
                # Checking if its a different dimension of placetype
                if "places" in og_st_fn:
                    if "NONNET20" not in starting_fn and "NONNET20"  in f:
                        print("continue")
                        continue
                    elif "NONNET50" not in starting_fn and "NONNET50"  in f:
                        print("continue")
                        continue
                fns_to_average.append(f)
        else:
            fns_to_average.append(f)
    """
    # Get an array of grouped filenames, where filenames are grouped if they are to be averaged
    # Determine this by checking if the only differentiator is the CSV number
    average_groups = []
    # For every FN
    for f in fns_to_average:
        # Remove the CSV part and then remake the string, find any matching strings with the same effect
        og_fn, fn = removeCSVText(f)
        average_group = [og_fn]
        for fn in fns_to_average:
            s_og_fn, s_fn = removeCSVText(fn)
            # If it matches without the CSV but isn't already added
            if s_fn == fn and s_og_fn not in average_group:
                average_group.append(s_og_fn)
        # Add to the collection
        if len(average_group) > 1:
            average_groups.append(average_group)
    average_fns = []
    # Average the CSV's, return filenames to add to the overall csv compilation
    for g in average_groups:
        for i in range(len(g)):
            g[i] = csv_folder_fn + g[i]
        average_fns.append(average_csv(g))
    """
    for i in range(len(fns_to_average)):
        fns_to_average[i] = csv_folder_fn + fns_to_average[i]
    return averageCSVs(fns_to_average)

