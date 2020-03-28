from glob import  glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import json
import argparse
import sys
import os

# constants
path_save_max_values = "./dict_max_matches.pkl"
# if output_mode = 3, save the pdfs (see below for more info)
path_save_p_dist_all = "./p_dist_all.pkl"
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


def handle_cmdline_args():
    """Return an object with attributes 'infile' and 'outfile', after
handling the command line arguments"""

    parser = argparse.ArgumentParser(
        description='Generate synthetic data from a specification in a json '
        'file using the "synth-method" described in the json file.  ')

    parser.add_argument(
        '-i', dest='infile', required=True,
        help='The input json file. Must contain a "synth-method" property')

    parser.add_argument(
        '-o', dest='outfile_prefix', required=True, help='The prefix of the output paths (data json and csv), relative to the QUIPP-pipeline root directory')

    args = parser.parse_args()
    return args

def compare_rows(row_check, dataframe_check, drop_column="idx"):
    """
    Find all the matched rows in dataframe_check given a row to check (row_check)
    """
    # all(1) means that all items of row_check should match with rows in dataframe_check
    # except for drop_column
    dataframe_matched = dataframe_check[(dataframe_check.drop(drop_column, axis=1) ==\
                                         row_check.drop(drop_column)).all(1)]
    return dataframe_matched

def main():
    # read command line options
    args = handle_cmdline_args()

    with open(args.infile) as f:
        synth_params = json.load(f)

    if not (synth_params["enabled"] and synth_params['parameters_disclosure_risk']['enabled']):
        return

    # read dataset name from .json
    dataset = synth_params["dataset"]
    synth_method = synth_params["synth-method"]
    path_released_ds = args.outfile_prefix
    if synth_method == 'sgf':
        path_original_ds = os.path.join(path_released_ds, os.path.basename(dataset) + "_numcat.csv")
    else:
        path_original_ds = os.path.abspath(dataset) + '.csv'

    # read parameters from .json
    parameters = synth_params["parameters"]
    disclosure_risk_parameters = synth_params["parameters_disclosure_risk"]

    # read original data set
    data_full = pd.read_csv(path_original_ds)

    # read/set intruder samples number
    if disclosure_risk_parameters['num_samples_intruder'] > data_full.shape[0]:
        sys.exit("Intruder samples cannot be more than original dataset samples: " + disclosure_risk_parameters['num_samples_intruder'] + " > " + data_full.shape[0])
    elif disclosure_risk_parameters['num_samples_intruder'] == -1:
        num_samples_intruder = data_full.shape[0]
    else:
        num_samples_intruder = disclosure_risk_parameters['num_samples_intruder']

    # sample indexes and use them to select rows from original data to form intruder dataset
    # also save indexes to .json
    np.random.seed(parameters['random_state'])
    indexes = np.random.choice(data_full.shape[0], num_samples_intruder, replace=False).tolist()
    data_intruder = data_full.loc[indexes, disclosure_risk_parameters['vars_intruder']]
    data_intruder.to_csv(path_released_ds + "/intruder_data.csv", index=False)
    with open(path_released_ds + "/intruder_indexes.json", 'w') as f:
        json.dump(indexes, f)

    # itdr: intruder
    df_itdr = pd.read_csv(path_released_ds + "/intruder_data.csv")
    # XXX should be changed after indices are added to real/synthetic data
    df_itdr["idx"] = df_itdr.index

    # list of paths of the released/synthetic datasets
    list_paths_released_ds = glob(path_released_ds + "/synthetic_data_*.csv")
    list_paths_released_ds.sort()

    # if output_mode == 1, set the threshold to 0.99,
    # this should help with floating point comparisons of numbers that are very close
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

    # save outputs
    with open(path_save_max_values, "wb") as output_file:
        pickle.dump(dict_max_matches, output_file)
    # output_mode == 3, returns full probability
    if output_mode == 3:
        with open(path_save_p_dist_all, "wb") as output_file:
            pickle.dump(p_dist_all, output_file)

    # Plot p.d.f computed in the previous step
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


    # Calculate privacy metrics
    with open(path_released_ds + "/intruder_indexes.json") as f_intruder_indexes:
        intruder_indexes = json.load(f_intruder_indexes)

    c = {key: len(value[0]) for key, value in dict_max_matches.items()}
    I = {key: np.multiply(intruder_indexes[int(key)] in value[0], 1) for key, value in dict_max_matches.items()}
    products = {k: c.get(k) * I.get(k) for k in set(c)}
    K = {key: np.multiply(value == 1, 1) for key, value in products.items()}
    c_indicator = {key: np.multiply(value == 1, 1) for key, value in c.items()}

    EMRi = sum({k: I.get(k) / c.get(k) for k in set(c)}.values())
    TMRi = float(sum(K.values()))
    TMRa = TMRi / sum(c_indicator.values())
    metrics = {'EMRi': EMRi, 'TMRi': TMRi, 'TMRa': TMRa}
    print(f"\nDisclosure risk metrics: {metrics}")

    with open(path_released_ds + "/privacy_metric_disclosure_risk.json", 'w') as f:
        json.dump(metrics, f)

if __name__=='__main__':
    main()