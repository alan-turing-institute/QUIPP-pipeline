#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""NOTE: Developed and tested with dython=0.5.1"""

import os
import argparse
import json
import numpy as np
import pandas as pd
from dython.nominal import cramers_v, theils_u, correlation_ratio


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


def cramers_v_corrected_matrix(df):
    """Wrapper around cramers_v, does the computation
    for all combinations of columns and fills an output matrix."""

    output = np.empty([df.shape[1], df.shape[1]])
    array = np.array(df)

    for i1 in range(array.shape[1]):
        for i2 in range(array.shape[1]):
            if i1 >= i2:
                output[i1, i2] = cramers_v(array[:, i1], array[:, i2])
                output[i2, i1] = output[i1, i2]

    return output


def theils_u_matrix(df):
    """Wrapper around theils_u, does the computation
    for all combinations of columns and fills an output matrix."""

    output = np.empty([df.shape[1], df.shape[1]])
    array = np.array(df)

    for i1 in range(array.shape[1]):
        for i2 in range(array.shape[1]):
            output[i1, i2] = theils_u(array[:, i1], array[:, i2])

    return output


def correlation_ratio_matrix(df, categorical, numeric):
    """Wrapper around correlation_ratio, does the computation
    for all combinations of discrete-continuous columns and fills an output matrix."""

    output = np.empty((df[categorical].shape[1], df[numeric].shape[1]))

    for i1, col1 in enumerate(df[categorical].columns):
        for i2, col2 in enumerate(df[numeric].columns):
            output[i1, i2] = correlation_ratio(np.array(df[col1]), np.array(df[col2]))

    return output


def utility_measure_correlations(synth_method, path_original_ds, path_original_meta, path_released_ds,
                                 output_file_json, correlation_parameters):
    """Calculated correlation and correlation-like metrics for all combinations of columns and print and
    saves the results."""

    # Read metadata in JSON format
    with open(path_original_meta) as orig_metadata_json:
        orig_metadata = json.load(orig_metadata_json)

    # Divide columns into discrete and numeric,
    # Discrete columns will be later vectorized
    categorical_features = []
    numeric_features = []
    for col in orig_metadata['columns']:
        if synth_method == 'sgf':
            categorical_features.append(col["name"])
        elif col['type'] in ['Categorical', 'Ordinal', "DateTime"]:
            categorical_features.append(col["name"])
        else:
            numeric_features.append(col["name"])

    # Read original and released/synthetic datasets
    # NOTE: Only the first synthetic data set is used for utility evaluation
    orig_df = pd.read_csv(path_original_ds)
    rlsd_df = pd.read_csv(path_released_ds + "/synthetic_data_1.csv")

    print("[INFO] Utility measurements")

    # Calculate Cramer's V and Thiel's U for categorical-categorical combinations of columns
    if len(categorical_features) > 0:
        cramers_matrix_orig = cramers_v_corrected_matrix(orig_df[categorical_features])
        cramers_matrix_rlsd = cramers_v_corrected_matrix(rlsd_df[categorical_features])
        theils_matrix_orig = theils_u_matrix(orig_df[categorical_features])
        theils_matrix_rlsd = theils_u_matrix(rlsd_df[categorical_features])

    # Calculate correlations for continuous-continuous combinations of columns
    if len(numeric_features) > 0:
        correlation_matrix_orig = np.array(orig_df[numeric_features].corr())
        correlation_matrix_rlsd = np.array(rlsd_df[numeric_features].corr())

    # Calculate correlation ratio for continuous-categorical combinations of columns
    if (len(categorical_features) > 0) and (len(numeric_features) > 0):
        correlation_ratio_matrix_orig = correlation_ratio_matrix(orig_df, categorical_features, numeric_features)
        correlation_ratio_matrix_rlsd = correlation_ratio_matrix(rlsd_df, categorical_features, numeric_features)
    
    # Store in dictionary after converting numpy arrays to lists in order to allow .json serialisation
    utility_collector = {"Cramers_V_Original": cramers_matrix_orig.tolist(),
                         "Cramers_V_Synthetic": cramers_matrix_rlsd.tolist(),
                         "Theils_U_Original": theils_matrix_orig.tolist(),
                         "Theils_U_Synthetic": theils_matrix_rlsd.tolist(),
                         "Correlations_Original": correlation_matrix_orig.tolist(),
                         "Correlations_Synthetic": correlation_matrix_rlsd.tolist(),
                         "Correlation_Ratio_Original": correlation_ratio_matrix_orig.tolist(),
                         "Correlation_Ratio_Synthetic": correlation_ratio_matrix_rlsd.tolist()
                         }

    with open(output_file_json, "w") as out_fio:
        json.dump(utility_collector, out_fio)


def main():
    args = handle_cmdline_args()

    with open(args.infile) as f:
        synth_params = json.load(f)

    if not (synth_params["enabled"] and synth_params['parameters_correlations']['enabled']):
        return

    # read dataset name from .json
    dataset = synth_params["dataset"]
    synth_method = synth_params["synth-method"]
    path_original_meta = os.path.abspath(dataset) + '.json'
    path_released_ds = args.outfile_prefix
    if synth_method == 'sgf':
        path_original_ds = os.path.join(path_released_ds, os.path.basename(dataset) + "_numcat.csv")
    else:
        path_original_ds = os.path.abspath(dataset) + '.csv'

    # read parameters from .json
    correlations_parameters = synth_params["parameters_correlations"]

    output_file_json = path_released_ds + "/correlations.json"

    np.random.seed(synth_params['parameters']['random_state'])

    utility_measure_correlations(synth_method,
                                        path_original_ds,
                                        path_original_meta,
                                        path_released_ds,
                                        output_file_json,
                                        correlations_parameters)


if __name__ == '__main__':
    main()