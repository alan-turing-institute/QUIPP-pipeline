## prepare_data.py

import numpy as np
import pandas as pd
import json
import argparse
import sys
import os


def handle_cmdline_args():
    parser = argparse.ArgumentParser(
        description='Read input files and output file for run of sgf.')

    parser.add_argument("--input-filename", type=str, default='hospital_ae_data_deidentify',
                        help="Input data filename (no extension)")
    parser.add_argument("--output-filename", type=str, default='hospital_ae_data_deidentify_numerical_categories',
                        help="Output data filename (no extension)")
    parser.add_argument("--output-dir", type=str, default=os.getcwd(), help="Output directory")

    args = parser.parse_args()

    return args

def make_column_to_numerical_category(data, column):

    # map numbers with and categories
    dict_cat = {}
    data_cat = data[column].astype("category").cat.codes + 1
    dict_cat[column] = [{k+1: v for k, v in dict(enumerate(data[column].astype("category").cat.categories)).items()}]
    return data_cat, dict_cat


def main():
    args = handle_cmdline_args()

    # read original data
    data = pd.read_csv(args.input_filename + ".csv")

    dict_cats = []

    for column in data.columns:

        # only process columns with non numerical categories
        #if pd.api.types.is_integer_dtype(data[column]) == False:

            # rewrite the column on the dataframe, and get category-number map
        data[column], data_dict = make_column_to_numerical_category(data,column)
        dict_cats.append(data_dict)

    # saved processed data
    data_file = os.path.join(args.output_dir, args.output_filename) + ".csv"
    data.to_csv(data_file, index=False)
    print('deidentified dataset written out to: ' + data_file)

    # saved ategory-number map
    json_dict_file = os.path.join(args.output_dir, args.output_filename) + ".json"
    with open(json_dict_file, "w") as jsondict:
        json.dump(dict_cats, jsondict,sort_keys=True)

    print('category map written out to: ' + json_dict_file)

if __name__ == "__main__":
    main()
