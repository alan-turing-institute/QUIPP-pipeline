"""
Code to calculate correlation-like utility metrics
"""

import json
import os
import sys
import warnings
import numpy as np
import pandas as pd
from dython.nominal import cramers_v, theils_u, correlation_ratio

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, "utilities"))
from utils import handle_cmdline_args, extract_parameters, find_column_types


def cramers_v_corrected_matrix(df):
    """
    Wrapper around dython's cramers_v. Given a dataframe df
    containing only categorical data, it calculates Cramer's V
    for each combination of columns, filling an output matrix, and
    returns the output matrix.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe containing only categorical data.

    Returns
    -------
    output : numpy.array
        An array with number of rows and columns equal to the
        number of columns of df, which contains the Cramer's V
        metrics for all combinations of columns in df.
    """

    output = np.empty([df.shape[1], df.shape[1]])
    array = np.array(df)

    for i1 in range(array.shape[1]):
        for i2 in range(array.shape[1]):
            if i1 >= i2:
                output[i1, i2] = cramers_v(array[:, i1], array[:, i2])
                output[i2, i1] = output[i1, i2]

    return output


def theils_u_matrix(df):
    """
    Wrapper around dython's theils_u. Given a dataframe df
    containing only categorical data, it calculates Theil's U
    for each combination of columns, filling an output matrix, and
    returns the output matrix.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe containing only categorical data.

    Returns
    -------
    output : numpy.array
        An array with number of rows and columns equal to the
        number of columns of df, which contains the Theil's U
        metrics for all combinations of columns in df.
    """

    output = np.empty([df.shape[1], df.shape[1]])
    array = np.array(df)

    for i1 in range(array.shape[1]):
        for i2 in range(array.shape[1]):
            output[i1, i2] = theils_u(array[:, i1], array[:, i2])

    return output


def correlation_ratio_matrix(df, categorical, numeric):
    """
    Wrapper around dython's correlation ratio. Given a dataframe df
    containing a mix of categorical and continuous data, it calculates
    the correlation ratio for each combination of columns, filling an
    output matrix, and returns the output matrix.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe containing a mix of categorical and
        continuous data.
    categorical : list
        A list of column names which contain categorical data.
    numeric : list
        A list of column names which contain continuous data.

    Returns
    -------
    output : numpy.array
        An array with number of rows equal to the length of
        categorical and number of columns equal to the length
        of continuous, which contains the correlation ratios for
        all combinations of columns in df.
    """

    output = np.empty((df[categorical].shape[1], df[numeric].shape[1]))

    for i1, col1 in enumerate(df[categorical].columns):
        for i2, col2 in enumerate(df[numeric].columns):
            output[i1, i2] = correlation_ratio(np.array(df[col1]),
                                               np.array(df[col2]))

    return output


def correlation_metrics(synth_method, path_original_ds,
                        path_original_meta, path_released_ds,
                        output_file_json, random_seed=1234):
    """
    Calculates correlation and correlation-like metrics for all
    combinations of columns of the original and released datasets
    and saves the results into a .json file. These can be compared to
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

    print("[INFO] Calculating correlation-like utility metrics:")

    # set random seed
    np.random.seed(random_seed)

    # read metadata in JSON format
    with open(path_original_meta) as orig_metadata_json:
        orig_metadata = json.load(orig_metadata_json)

    # divide columns into categorical and numeric
    categorical_types = ['Categorical', 'Ordinal', 'DateTime']
    categorical_features, numeric_features = \
        find_column_types(orig_metadata, synth_method, categorical_types)

    # read original and released/synthetic datasets,
    # only the first synthetic data set (synthetic_data_1.csv)
    # is used for utility evaluation
    orig_df = pd.read_csv(path_original_ds)
    rlsd_df = pd.read_csv(path_released_ds + "/synthetic_data_1.csv")

    with warnings.catch_warnings(record=True) as warns:
        # calculate Cramer's V and Thiel's U for
        # categorical-categorical combinations of
        # columns for both datasets
        if len(categorical_features) > 0:
            print("Cramer's V...")
            cramers_matrix_orig = cramers_v_corrected_matrix(orig_df[categorical_features])
            cramers_matrix_rlsd = cramers_v_corrected_matrix(rlsd_df[categorical_features])
            print("Theil's U...")
            theils_matrix_orig = theils_u_matrix(orig_df[categorical_features])
            theils_matrix_rlsd = theils_u_matrix(rlsd_df[categorical_features])

        # calculate correlations for continuous-continuous
        # combinations of columns
        if len(numeric_features) > 0:
            print("Correlation...")
            correlation_matrix_orig = np.array(orig_df[numeric_features].corr())
            correlation_matrix_rlsd = np.array(rlsd_df[numeric_features].corr())

        # calculate correlation ratio for
        # continuous-categorical combinations of columns
        if (len(categorical_features) > 0) and (len(numeric_features) > 0):
            print("Correlation ratio...")
            correlation_ratio_matrix_orig = correlation_ratio_matrix(orig_df,
                                                                     categorical_features,
                                                                     numeric_features)
            correlation_ratio_matrix_rlsd = correlation_ratio_matrix(rlsd_df,
                                                                     categorical_features,
                                                                     numeric_features)
    
    # store in dictionary after converting numpy arrays
    # to lists in order to allow .json serialisation
    utility_collector = {"Cramers_V_Original": cramers_matrix_orig.tolist(),
                         "Cramers_V_Released": cramers_matrix_rlsd.tolist(),
                         "Theils_U_Original": theils_matrix_orig.tolist(),
                         "Theils_U_Released": theils_matrix_rlsd.tolist(),
                         "Correlations_Original": correlation_matrix_orig.tolist(),
                         "Correlations_Released": correlation_matrix_rlsd.tolist(),
                         "Correlation_Ratio_Original": correlation_ratio_matrix_orig.tolist(),
                         "Correlation_Ratio_Released": correlation_ratio_matrix_rlsd.tolist()
                         }

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
    # correlation utility metrics are not enabled, stop here
    if not (synth_params["enabled"] and
            synth_params[f'utility_parameters_correlations']['enabled']):
        return

    # extract paths and other parameters from args
    synth_method, path_original_ds, \
    path_original_meta, path_released_ds, \
    random_seed = extract_parameters(args, synth_params)

    # create output .json full path
    output_file_json = path_released_ds + f"/utility_correlations.json"

    # calculate and save correlation-like metrics
    correlation_metrics(synth_method, path_original_ds,
                        path_original_meta, path_released_ds,
                        output_file_json, random_seed)


if __name__ == '__main__':
    main()
