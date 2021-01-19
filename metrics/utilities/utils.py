"""
Utility methods used in other parts of the code.
"""

import argparse
import json
import os


def handle_cmdline_args():
    """
    Return an object with attributes 'infile' and
    'outfile', after handling the command line arguments
    """

    parser = argparse.ArgumentParser(
        description='Generate synthetic data from a specification in a json '
                    'file using the "synth-method" described in the json file.  ')

    parser.add_argument(
        '-i', dest='infile', required=True,
        help='The input json file. Must contain a "synth-method" property')

    parser.add_argument(
        '-o', dest='outfile_prefix', required=True,
        help='The prefix of the output paths (data json and csv), '
             'relative to the QUIPP-pipeline root directory')

    args = parser.parse_args()
    return args


def handle_cmdline_args_rf_seeds():
    """
    Return an object with attributes 'infile', 'outfile' and
    'seeds' after handling the command line arguments
    """

    parser = argparse.ArgumentParser(
        description='Generate synthetic data from a specification in a json '
                    'file using the "synth-method" described in the json file.  ')

    parser.add_argument(
        '-i', dest='infile', required=True,
        help='The input json file. Must contain a "synth-method" property')

    parser.add_argument(
        '-o', dest='outfile_prefix', required=True,
        help='The prefix of the output paths (data json and csv), '
             'relative to the QUIPP-pipeline root directory')

    parser.add_argument(
        '-s', dest='seeds', required=True,
        help='List of seeds to use for RF training')

    args = parser.parse_args()
    return args


def extract_parameters(args, synth_params):
    """
    This method takes args from the command line and
    a synth_params dictionary and extracts some key
    parameters needed for running utility metric methods.
    These are file paths, the synthesis method and the
    random seed used.
    """

    # create dataset paths from .json contents
    dataset = synth_params["dataset"]
    synth_method = synth_params["synth-method"]
    path_original_meta = os.path.abspath(dataset) + '.json'
    path_released_ds = args.outfile_prefix
    if synth_method == 'sgf':
        path_original_ds = os.path.join(path_released_ds,
                                        os.path.basename(dataset) + "_numcat.csv")
    else:
        path_original_ds = os.path.abspath(dataset) + '.csv'

    random_seed = synth_params['parameters']['random_state']

    return synth_method, path_original_ds, path_original_meta, \
           path_released_ds, random_seed


def find_column_types(orig_metadata, synth_method, categorical_types):
    """
    This method creates a list of categorical columns (defined by user)
    and a list of numerical columns, using types included in the
    .json metadata.
    """
    categorical_features = []
    numeric_features = []
    for col in orig_metadata['columns']:
        # sgf works only with categorical features
        if synth_method == 'sgf':
            categorical_features.append(col["name"])
        elif col['type'] in categorical_types:
            categorical_features.append(col["name"])
        else:
            if col['type'] not in ['String', 'DateTime']:
                numeric_features.append(col["name"])

    return categorical_features, numeric_features
