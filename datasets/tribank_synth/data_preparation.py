# Simple code to prepare Tribank data for QUIPP

import pandas as pd

path2csv = 'path/to/csv/file'
num_samples = 1000
sel_cols = ["payer", "receiver", "amount"]
output_filename = "./tribank_synth.csv"

# read csv file
df_rd = pd.read_csv(path2csv, sep='\t')

# select columns
df_pairs_sel_cols = df_rd[sel_cols].copy()

# sample
df_pairs_sel_cols = df_pairs_sel_cols.sample(num_samples)  

# output
df_pairs_sel_cols.to_csv(output_filename, index=False)
