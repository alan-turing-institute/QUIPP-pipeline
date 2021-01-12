"""
Code to calculate feature importance utility metrics
"""

import featuretools as ft
import json
import os
import sys
import warnings
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, "utilities"))
from utils import handle_cmdline_args, extract_parameters, find_column_types


def feature_importance_metrics(synth_method, path_original_ds,
                        path_original_meta, path_released_ds,
                        output_file_json, random_seed=1234):
    """
    Calculates feature importance differences between the original
    and released datasets, using a random forest classification model.
    Saves the results into a .json file. These can be compared to
    estimate the utility of the released dataset.

    Parameters
    ----------
    synth_method: string
        The synthesis method used to create the released dataset.
    path_original_ds : string
        Path to the original dataset.
    path_original_meta : string
        Path to the original metadata.
    path_released_ds : string
        Path to the released dataset.
    output_file_json : string
        Path to the output json file that will be generated.
    random_seed : integer
        Random seed for numpy. Defaults to 1234
    """

    print("[INFO] Calculating feature importance utility metrics:")

    # set random seed
    np.random.seed(random_seed)

    # read metadata in JSON format
    with open(path_original_meta) as orig_metadata_json:
        orig_metadata = json.load(orig_metadata_json)

    # read original and released/synthetic datasets,
    # only the first synthetic data set (synthetic_data_1.csv)
    # is used for utility evaluation
    orig_df = pd.read_csv(path_original_ds)
    rlsd_df = pd.read_csv(path_released_ds + "/synthetic_data_1.csv")

    with warnings.catch_warnings(record=True) as warns:
        # calculate metric...

    # store metrics in dictionary
    utility_collector = {}

    # print warnings
    if len(warns) > 0:
        print("WARNINGS:")
        for iw in warns:
            print(iw.message)

    # save as .json
    with open(output_file_json, "w") as out_fio:
        json.dump(utility_collector, out_fio)


def main():
    # process command line arguments
    args = handle_cmdline_args()

    # read run input parameters file
    with open(args.infile) as f:
        synth_params = json.load(f)

    # if the whole .json is not enabled or if the
    # feature importance utility metrics are not enabled, stop here
    if not (synth_params["enabled"] and
            synth_params[f'utility_parameters_feature_importance']['enabled']):
        return

    # extract paths and other parameters from args
    synth_method, path_original_ds, \
    path_original_meta, path_released_ds, \
    random_seed = extract_parameters(args, synth_params)

    # create output .json full path
    output_file_json = path_released_ds + f"/utility_feature_importance.json"

    # calculate and save feature importance metrics
    feature_importance_metrics(synth_method, path_original_ds,
                               path_original_meta, path_released_ds,
                               output_file_json, random_seed)


if __name__ == '__main__':
    main()
