#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from glob import  glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

# --- inputs
# path to released (synthetic) datasets 
path_released_ds = "../synth-output/synthpop-example-2/synthetic_data_*.csv"
# path to intruder's dataset
path_intruder_ds = "../synth-output/synthpop-example-2/intruder_data.csv"
# save maximum values/indices for all intruder's rows in a dictionary
path_save_max_values = "./dict_max_matches.pkl"
# if output_mode = 3, save the pdfs (see below for more info)
path_save_p_dist_all = "./p_dist_all.pkl"
# XXX REMOVE? 
# column that contains indices of the original (not synthesized) dataset
# when matching rows between two datasets, ignore indx_column 
indx_column = "idx"
# mode 1: only return the maximum
#    this is mode is idential to mode 2 with threshold_max = 0.999
# mode 2: return all values more than np.max(p.d.f of one intruder row)*threshold_max
# mode 3: return full probability distribution for each intruder row 
#    output probability distribution over each row of the released data, 
#    e.g., if the released data has 9000 rows and the intruder's data has 1000 rows, 
#    a numpy array with 1000 x 9000 dimensions will be created.
output_mode = 1
# the following value will be used to extract "found" rows from the released data:
# np.max(p.d.f of one intruder row)*threshold_max
threshold_max = 0.8
# ---

# -------- compare_rows function
def compare_rows(row_check, dataframe_check, drop_column="idx"):
    """
    Find all the matched rows in dataframe_check given a row to check (row_check)
    """
    # all(1) means that all items of row_check should match with rows in dataframe_check
    # except for drop_column
    dataframe_matched = dataframe_check[(dataframe_check.drop(drop_column, axis=1) ==\
                                         row_check.drop(drop_column)).all(1)]
    return dataframe_matched

# -------- MAIN 
# itdr: intruder
df_itdr = pd.read_csv(path_intruder_ds)
# XXX should be changed after indices are added to real/synthetic data
df_itdr["idx"] = df_itdr.index

# list of paths of the released datasets
list_paths_released_ds = glob(path_released_ds)
list_paths_released_ds.sort()

# if output_mode == 1, set the threshold to 0.99,
# this should help with floating point comparison 
if output_mode == 1:
    threshold_max = 0.999

dict_matches = {}
num_rows_released = False
num_files_released = False
# rlsd: released
# itdr: intruder
print(10*"===")
print("[INFO] Find similar rows between released and intruder's datasets.\n")
for i_rlsd, one_released_ds in enumerate(list_paths_released_ds):
    print(f"Processing {one_released_ds}")
    df_rlsd = pd.read_csv(one_released_ds)
    if not num_rows_released:
        num_rows_released = len(df_rlsd)
        num_files_released = len(list_paths_released_ds)

    # XXX should be changed after indices are added to real/synthetic data
    df_rlsd["idx"] = df_rlsd.index

    # consider only columns that intruder has access to
    df_rlsd_cols_selected = df_rlsd[df_itdr.columns]
    for i_itdr, one_intruder_row in df_itdr.iterrows():
        row_matches = compare_rows(one_intruder_row, 
                                   df_rlsd_cols_selected, 
                                   drop_column=indx_column)
        matches_num_rows = len(row_matches)
        if matches_num_rows > 0:
            matches_indx_list = row_matches.idx.to_list()
        else:
            matches_indx_list = []

        if not f"{i_itdr}" in dict_matches.keys():
            dict_matches[f"{i_itdr}"] = [matches_indx_list]
        else:
            dict_matches[f"{i_itdr}"].append(matches_indx_list)
print(10*"===")

print("\n[INFO] Create probability distributions for each row in intruder's dataset.\n")
p_dist_all = np.array([])
dict_max_matches = {}
for i_itdr in dict_matches:
    print(".", end="", flush=True)
    # first create a zero array with num_rows_released as the number of entries
    p_dist_row = np.zeros(num_rows_released)
    for m_rlsd in range(num_files_released):
        indicator_row = dict_matches[i_itdr][m_rlsd] 
        len_indicator_row = len(indicator_row)
        # Part of equation 6 in "Accounting for Intruder Uncertainty Due to Sampling When Estimating Identification Disclosure Risks in Partially Synthetic Data" paper.
        p_dist_row[indicator_row] += np.ones(len_indicator_row)/len_indicator_row
    # normalize based on the number of released datasets
    p_dist_row /= float(num_files_released)

    # output_mode == 3, returns full probability
    if output_mode == 3:
        if len(p_dist_all) == 0:
            p_dist_all = np.vstack([p_dist_row])
        else:
            p_dist_all = np.vstack([p_dist_all, p_dist_row])
    
    # store indices and values correspond to p_dist_row >= (np.max(p_dist_row)*threshold_max) 
    indx_max_matches = np.where(p_dist_row >= (np.max(p_dist_row)*threshold_max))[0].tolist()
    values_max_matches = p_dist_row[indx_max_matches].tolist()
    dict_max_matches[f"{i_itdr}"] = [indx_max_matches, values_max_matches]

# -------- save outputs 
with open(path_save_max_values, "wb") as output_file:
    pickle.dump(dict_max_matches, output_file)
# output_mode == 3, returns full probability
if output_mode == 3:
    with open(path_save_p_dist_all, "wb") as output_file:
        pickle.dump(p_dist_all, output_file)

# -------- Plot p.d.f computed in the previous step 
# This only works with output_mode == 3 (return full probability)
row_select = 0
while (row_select >= 0) and (output_mode == 3):
    row_select = int(input("\n\nSelect a row (indexed from 0) in the intruder's dataset. (or enter -1 to exit)  "))
    if row_select < 0:
        break
    elif row_select >= len(df_itdr):
        print(f"[ERROR] total number of rows in the intruder's dataset: {len(df_itdr)}")
        continue

    # print the selected row in dict_matches 
    print(dict_matches[f"{row_select}"])

    # plot the p.d.f.
    plt.figure(figsize=(12, 6))
    plt.plot(p_dist_all[row_select, :].T, c="k")
    plt.xlabel("Released data row", size=22)
    plt.ylabel("p.d.f", size=22)
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.title(f"Intruder row: {row_select}", size=24)
    plt.grid()
    plt.tight_layout()
    plt.show()