#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Utility metrics using scikit-learn library.
"""

import argparse
import json
import numpy as np
import pandas as pd
import warnings
import os
import sys

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
# --- Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def utility_measure_sklearn_classifiers(path_original_ds, path_original_meta, path_released_ds, 
                                        input_columns, label_column, test_train_ratio, classifiers,
                                        output_file_json, num_leaked_rows, random_seed=1364):
    np.random.seed(random_seed)

    # Read metadata in JSON format
    with open(path_original_meta) as orig_metadata_json:
        orig_metadata = json.load(orig_metadata_json)

    # Divide columns into discrete and numeric,
    # Discrete columns will be later vectorized
    discrete_features = []
    numeric_features = []
    for col in orig_metadata['columns']:
        if col['type'] in ['Categorical', 'Ordinal', 'DiscreteNumerical', "DateTime"]:
            discrete_features.append(col["name"])
        else:
            numeric_features.append(col["name"])

    # Read original and released/synthetic datasets
    # NOTE: Only the first synthetic data set is used for utility evaluation
    orig_df = pd.read_csv(path_original_ds)
    rlsd_df = pd.read_csv(path_released_ds + "/synthetic_data_1.csv")
    if num_leaked_rows > 0:
        rlsd_df[:num_leaked_rows] = orig_df[:num_leaked_rows]

    # Drop nans
    orig_df = orig_df.dropna(axis=0)
    rlsd_df = rlsd_df.dropna(axis=0)

    # split original
    X_o, y_o = orig_df[input_columns], orig_df[label_column]
    X_train_o, X_test_o, y_train_o, y_test_o = \
        train_test_split(X_o, y_o, test_size=test_train_ratio, 
                         random_state=random_seed)

    # split released
    X_r, y_r = rlsd_df[input_columns], rlsd_df[label_column]
    X_train_r, X_test_r, y_train_r, y_test_r = \
        train_test_split(X_r, y_r, test_size=test_train_ratio,
                         random_state=random_seed)

    # Create preprocessing pipelines for both numeric and discrete data.
    # SimpleImputer: Imputation transformer for completing missing values.
    # StandardScaler: Standardize features by removing the mean and scaling to unit variance
    numeric_transformer = Pipeline(steps=[
        #('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
        ])

    # OneHotEncoder: Encode discrete features as a one-hot numeric array.
    discrete_transformer = Pipeline(steps=[
        #('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(sparse=False, handle_unknown='ignore'))
        ])

    # extract numeric/discrete features used in dataframes
    numeric_features_in_df = list(set(numeric_features).intersection(input_columns))
    discrete_features_in_df = list(set(discrete_features).intersection(input_columns))
    # column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features_in_df),
            ('cat', discrete_transformer, discrete_features_in_df)
            ])

    utility_collector = {}
    print("[INFO] Utility measurements")
    print("\nThree values for each metric:\nmodel trained on original tested on original / model trained on released tested on original / model trained on released tested on released\n")
    print(30*"-----")
    for one_clf in classifiers:
        with warnings.catch_warnings(record=True) as warns:
            # original dataset
            # Append classifier to preprocessing pipeline.
            clf_orig = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('classifier', one_clf(**classifiers[one_clf]))])
            clf_orig.fit(X_train_o, y_train_o)
            # o ---> o
            y_test_pred_o_o = clf_orig.predict(X_test_o)

            # released dataset
            clf_rlsd = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('classifier', one_clf(**classifiers[one_clf]))])
            clf_rlsd.fit(X_train_r, y_train_r)
            # r ---> o
            y_test_pred_r_o = clf_rlsd.predict(X_test_o)
            # r ---> r
            y_test_pred_r_r = clf_rlsd.predict(X_test_r)

            clf_name = one_clf.__name__
            utility_collector[clf_name] = \
                    {
                        "accu_o_o": accuracy_score(y_test_pred_o_o, y_test_o)*100., 
                        "prec_o_o": precision_score(y_test_pred_o_o, y_test_o, average='weighted', zero_division=True)*100.,
                        "reca_o_o": recall_score(y_test_pred_o_o, y_test_o, average='weighted', zero_division=True)*100.,
                        "f1_o_o": f1_score(y_test_pred_o_o, y_test_o, average='weighted', zero_division=True)*100.,

                        "accu_r_o": accuracy_score(y_test_pred_r_o, y_test_o)*100., 
                        "prec_r_o": precision_score(y_test_pred_r_o, y_test_o, average='weighted', zero_division=True)*100.,
                        "reca_r_o": recall_score(y_test_pred_r_o, y_test_o, average='weighted', zero_division=True)*100.,
                        "f1_r_o": f1_score(y_test_pred_r_o, y_test_o, average='weighted', zero_division=True)*100.,

                        "accu_r_r": accuracy_score(y_test_pred_r_r, y_test_r)*100., 
                        "prec_r_r": precision_score(y_test_pred_r_r, y_test_r, average='weighted', zero_division=True)*100.,
                        "reca_r_r": recall_score(y_test_pred_r_r, y_test_r, average='weighted', zero_division=True)*100.,
                        "f1_r_r": f1_score(y_test_pred_r_r, y_test_r, average='weighted', zero_division=True)*100.,
                    }

            print(f"{clf_name:30}, \
            accu: {utility_collector[clf_name]['accu_o_o']:6.02f}/{utility_collector[clf_name]['accu_r_o']:6.02f}/{utility_collector[clf_name]['accu_r_r']:6.02f} \
            prec: {utility_collector[clf_name]['prec_o_o']:6.02f}/{utility_collector[clf_name]['prec_r_o']:6.02f}/{utility_collector[clf_name]['prec_r_r']:6.02f} \
            reca: {utility_collector[clf_name]['reca_o_o']:6.02f}/{utility_collector[clf_name]['reca_r_o']:6.02f}/{utility_collector[clf_name]['reca_r_r']:6.02f} \
            F1: {utility_collector[clf_name]['f1_o_o']:6.02f}/{utility_collector[clf_name]['f1_r_o']:6.02f}/{utility_collector[clf_name]['f1_r_r']:6.02f} \
                ")

    print(30*"-----")
    print("WARNINGS:")
    for iw in warns: print(iw.message)

    with open(output_file_json, "w") as out_fio:
        json.dump(utility_collector, out_fio)

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

def main():
    args = handle_cmdline_args()

    with open(args.infile) as f:
        synth_params = json.load(f)

    if not (synth_params["enabled"] and synth_params['parameters_sklearn_utility']['enabled']):
        return

    # read dataset name from .json
    dataset = synth_params["dataset"]
    path_original_ds = os.path.abspath(dataset) + '.csv'
    path_original_meta = os.path.abspath(dataset) + '.json'
    path_released_ds = args.outfile_prefix

    # read parameters from .json
    #parameters = synth_params["parameters"]
    sklearn_utility_parameters = synth_params["parameters_sklearn_utility"]

    input_columns = sklearn_utility_parameters["input_columns"]
    label_column = sklearn_utility_parameters["label_column"]
    test_train_ratio = sklearn_utility_parameters["test_train_ratio"]
    output_file_json = path_released_ds + "/utility_metric_sklearn.json"
    num_leaked_rows = sklearn_utility_parameters["num_leaked_rows"]

    print("\n=================================================")
    print("[WARNING] the classifiers are hardcoded in main()")
    print("=================================================\n")
    # List of classifiers and their arguments
    classifiers = {
        LogisticRegression: {"max_iter": 10000},
        KNeighborsClassifier: {"n_neighbors": 3},
        SVC: {"kernel": "linear", "C": 0.025},
        # SVC: {"gamma": 2, "C": 1},
        # GaussianProcessClassifier: {"kernel": 1.0 * RBF(1.0)},
        # DecisionTreeClassifier: {"max_depth": 5},
        # RandomForestClassifier: {"max_depth": 5, "n_estimators": 10, "max_features": 1},
        # MLPClassifier: {"alpha": 1, "max_iter": 5000},
        # AdaBoostClassifier: {},
        GaussianNB: {},
        QuadraticDiscriminantAnalysis: {}
        }

    utility_measure_sklearn_classifiers(path_original_ds, 
                                        path_original_meta, 
                                        path_released_ds, 
                                        input_columns, 
                                        label_column, 
                                        test_train_ratio, 
                                        classifiers,
                                        output_file_json,
                                        num_leaked_rows)

if __name__=='__main__':
    main()
