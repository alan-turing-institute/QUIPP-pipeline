#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from glob import  glob
import pandas as pd

# --- inputs
path_released_ds = "../synth-output/synthpop-example-2/synthetic_data_*.csv"
path_intruder_ds = "../synth-output/synthpop-example-2/intruder_data.csv"

def compare_rows(row_check, dataframe_check):
    """
    Find all the matched rows in dataframe_check given a row to check (row_check)
    """
    # all(1) means that all items of row_check should match with rows in dataframe_check
    dataframe_matched = dataframe_check[(dataframe_check == row_check).all(1)]
    return dataframe_matched

df_itdr = pd.read_csv(path_intruder_ds)
# this dataframe contains the number of matches between each row of df_itdr and each released dataset
df_number_matches = pd.DataFrame()

list_paths_released_ds = glob(path_released_ds)
list_paths_released_ds.sort()

for i_rlsd, one_released_ds in enumerate(list_paths_released_ds):
    print(f"Processing {one_released_ds}")
    df_rlsd = pd.read_csv(one_released_ds)
    df_rlsd_cols_selected = df_rlsd[df_itdr.columns]
    
    for i_itdr, one_intruder_row in df_itdr.iterrows():
        row_matches = compare_rows(one_intruder_row, df_rlsd_cols_selected)
        if not f"matches_synth_{i_rlsd}" in df_number_matches:
            df_number_matches[f"matches_synth_{i_rlsd}"] = [None]*len(df_itdr)
        df_number_matches[f"matches_synth_{i_rlsd}"].iloc[i_itdr] = len(row_matches)

df_itdr = pd.concat([df_itdr, df_number_matches], axis=1)
import ipdb; ipdb.set_trace()
