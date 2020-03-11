#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import pickle
from glob import  glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- inputs
path_original_ds = "../generator-outputs/odi-nhs-ae/hospital_ae_data_deidentify.csv"
path_released_ds = "../synth-output/synthpop-example-2/synthetic_data_*.csv"
path_intruder_ds = "../synth-output/synthpop-example-2/intruder_data.csv"
path_save_max_values = "./dict_max_matches.pkl"
path_save_p_dist_all = "./p_dist_all.pkl"
indx_column = "idx"
# ---

def compare_rows(row_check, dataframe_check, drop_column="idx"):
    """
    Find all the matched rows in dataframe_check given a row to check (row_check)
    """
    # all(1) means that all items of row_check should match with rows in dataframe_check
    dataframe_matched = dataframe_check[(dataframe_check.drop(drop_column, axis=1) == row_check.drop(drop_column)).all(1)]
    return dataframe_matched

df_orig = pd.read_csv(path_original_ds)
df_itdr = pd.read_csv(path_intruder_ds)
# this dataframe contains the number of matches between each row of df_itdr and each released dataset
df_number_matches = pd.DataFrame()

# XXX should be changed after indices are added to real/synthetic data
df_orig["idx"] = df_orig.index
df_itdr["idx"] = df_itdr.index

list_paths_released_ds = glob(path_released_ds)
list_paths_released_ds.sort()

dict_matches = {}
dict_number_matches = {}
num_rows_released = False
num_files_released = False
for i_rlsd, one_released_ds in enumerate(list_paths_released_ds):
    print(f"Processing {one_released_ds}")
    df_rlsd = pd.read_csv(one_released_ds)
    if not num_rows_released:
        num_rows_released = len(df_rlsd)
        num_files_released = len(list_paths_released_ds)

    # XXX should be changed after indices are added to real/synthetic data
    df_rlsd["idx"] = df_rlsd.index

    df_rlsd_cols_selected = df_rlsd[df_itdr.columns]
    for i_itdr, one_intruder_row in df_itdr.iterrows():
        row_matches = compare_rows(one_intruder_row, df_rlsd_cols_selected, drop_column=indx_column)
        matches_num_rows = len(row_matches)
        if matches_num_rows > 0:
            matches_indx_list = row_matches.idx.to_list()
        else:
            matches_indx_list = []

        if not f"{i_itdr}" in dict_matches.keys():
            dict_matches[f"{i_itdr}"] = [matches_indx_list]
            dict_number_matches[f"{i_itdr}"] = [matches_num_rows]
        else:
            dict_matches[f"{i_itdr}"].append(matches_indx_list)
            dict_number_matches[f"{i_itdr}"].append(matches_num_rows)

print("\n[INFO] start creating probability distributions for each row in intruder's dataset.\n")

full_probability = True
p_dist_all = np.array([])
dict_max_matches = {}
threshold_max = 0.8
for i_itdr in dict_matches:
    print(".", end="", flush=True)
    p_dist_row = np.zeros(num_rows_released)
    for m_rlsd in range(num_files_released):
        indicator_row = dict_matches[i_itdr][m_rlsd] 
        len_indicator_row = len(indicator_row)
        p_dist_row[indicator_row] += np.ones(len_indicator_row)/len_indicator_row
    p_dist_row /= float(num_files_released)

    if full_probability:
        if len(p_dist_all) == 0:
            p_dist_all = np.vstack([p_dist_row])
        else:
            p_dist_all = np.vstack([p_dist_all, p_dist_row])
    indx_max_matches = np.where(p_dist_row >= (np.max(p_dist_row)*threshold_max))[0].tolist()
    values_max_matches = p_dist_row[indx_max_matches].tolist()
    dict_max_matches[f"{i_itdr}"] = [indx_max_matches, values_max_matches]

with open(path_save_max_values, "wb") as output_file:
    pickle.dump(dict_max_matches, output_file)

if full_probability:
    with open(path_save_p_dist_all, "wb") as output_file:
        pickle.dump(p_dist_all, output_file)

row_select = 0
while (row_select >= 0) and full_probability:
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