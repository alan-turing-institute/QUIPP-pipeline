#!/usr/bin/env python

"""
Utility metrics using scikit-learn library.
"""

import argparse
import codecs
import json
import numpy as np
import pandas as pd
import random
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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

import warnings

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
np.random.default_rng(42)


def utility_measure_sklearn_classifiers(synth_method, path_original_ds, path_original_meta, path_released_ds,
                                        input_columns, label_column, test_train_ratio, classifiers,
                                        output_file_json, num_leaked_rows, 
                                        disable_all_warnings=False, random_seed=1364):
    # Set random seeds for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    np.random.default_rng(random_seed)

    if disable_all_warnings:
        if not sys.warnoptions:
            warnings.simplefilter("ignore")
            os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

    # Read metadata in JSON format
    with open(path_original_meta) as orig_metadata_json:
        orig_metadata = json.load(orig_metadata_json)

    # Divide columns into discrete and numeric,
    # Discrete columns will be later vectorized
    discrete_features = []
    numeric_features = []
    for col in orig_metadata['columns']:
        if synth_method == 'sgf':
            discrete_features.append(col["name"])
        elif col['type'] in ['Categorical', 'Ordinal', 'DiscreteNumerical', "DateTime"]:
            discrete_features.append(col["name"])
        else:
            numeric_features.append(col["name"])

    # Read original and released/synthetic datasets
    # NOTE: Only the first synthetic data set is used for utility evaluation
    orig_df = pd.read_csv(path_original_ds)
    # XXX this should be a flag, fill all NaNs
    orig_df.fillna(orig_df.median(), inplace=True)
    rlsd_df = pd.read_csv(path_released_ds + "/synthetic_data_1.csv")
    # XXX this should be a flag, fill all NaNs
    rlsd_df.fillna(rlsd_df.median(), inplace=True)
    if num_leaked_rows > 0:
        rlsd_df[:num_leaked_rows] = orig_df[:num_leaked_rows]

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

    #import ipdb; ipdb.set_trace()

    # Create preprocessing pipelines for both numeric and discrete data.
    # SimpleImputer: Imputation transformer for completing missing values.
    # StandardScaler: Standardize features by removing the mean and scaling to unit variance
    numeric_transformer = Pipeline(steps=[
        #('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # OneHotEncoder: Encode discrete features as a one-hot numeric array.
    discrete_transformer = Pipeline(steps=[
        #('imputer', SimpleImputer(strategy='most_frequent')),
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

    utility_o_o = {}
    utility_r_o = {}
    utility_diff = {}
    utility_confusion_o_o = {}
    utility_confusion_r_o = {}
    print("[INFO] Utility measurements")
    print(
        "\nThree values for each metric:\nmodel trained on original tested on original / model trained on released tested on original / model trained on released tested on released\n")
    print(30 * "-----")
    for one_clf in classifiers:
        with warnings.catch_warnings(record=True) as warns:
            # original dataset
            # Append classifier to preprocessing pipeline.
            if classifiers[one_clf]["mode"] == "main":
                parameters = classifiers[one_clf]["params_main"]
                clf_orig = Pipeline(steps=[('preprocessor', preprocessor),
                                           ('classifier', one_clf(**parameters))])
            else:
                parameters = classifiers[one_clf]["params_range"]
                clf_orig = Pipeline(steps=[('preprocessor', preprocessor),
                                           ('classifier', one_clf())])
                clf_orig = GridSearchCV(clf_orig, parameters, scoring="f1_macro", n_jobs=-1)


            ### XXX
            ### # To report the results:
            #from sklearn.metrics import classification_report
            #classification_report(y_test_pred_o_o, y_test_o, output_dict=True)

            clf_orig.fit(X_train_o, y_train_o)

            # o ---> o
            y_test_pred_o_o = clf_orig.predict(X_test_o)

            # released dataset
            if classifiers[one_clf]["mode"] == "main":
                clf_rlsd = Pipeline(steps=[('preprocessor', preprocessor),
                                           ('classifier', one_clf(**parameters))])
            else:
                parameters_rlsd = {k.split("classifier__")[1]: v for k, v in clf_orig.best_params_.items()}
                clf_rlsd = Pipeline(steps=[('preprocessor', preprocessor),
                                           ('classifier', one_clf(**parameters_rlsd))])


            clf_rlsd.fit(X_train_r, y_train_r)
            # r ---> o
            y_test_pred_r_o = clf_rlsd.predict(X_test_o)
            # r ---> r
            y_test_pred_r_r = clf_rlsd.predict(X_test_r)

            clf_name = one_clf.__name__

            utility_o_o[clf_name] = calc_metrics(y_test_pred_o_o, y_test_o)
            utility_r_o[clf_name] = calc_metrics(y_test_pred_r_o, y_test_o)
            utility_diff[clf_name] = calc_diff_metrics(utility_o_o[clf_name], utility_r_o[clf_name])
            utility_confusion_o_o[clf_name] = calc_confusion_matrix(y_test_pred_o_o, y_test_o, 
                                                                    target_names=clf_orig.classes_)
            utility_confusion_r_o[clf_name] = calc_confusion_matrix(y_test_pred_r_o, y_test_o, 
                                                                    target_names=clf_rlsd.classes_)
    
    utility_overall_diff = calc_overall_diff(utility_diff)

    printMetric(utility_o_o, title="Trained on original and tested on original")
    printMetric(utility_r_o, title="Trained on released and tested on original")
    printSummary(utility_overall_diff, title="Overall difference")

    saveJson(utility_overall_diff, filename="utility_overall_diff.json", par_dir=path_released_ds)
    saveJson(utility_diff, filename="utility_diff.json", par_dir=path_released_ds)
    saveJson(utility_o_o, filename="utility_o_o.json", par_dir=path_released_ds)
    saveJson(utility_r_o, filename="utility_r_o.json", par_dir=path_released_ds)
    saveJson(utility_confusion_o_o, filename="utility_confusion_o_o.json", par_dir=path_released_ds)
    saveJson(utility_confusion_r_o, filename="utility_confusion_r_o.json", par_dir=path_released_ds)

    print(30 * "-----")
    print("WARNINGS:")
    for iw in warns: print(iw.message)
    print()


# ======== Functions

def calc_confusion_matrix(y_pred, y_test, target_names):
    output = {}
    output["conf_matrix"] = confusion_matrix(y_pred, y_test).tolist()
    output["target_names"] = target_names.tolist()
    return output


def saveJson(inp_dict, filename, par_dir):
    if not os.path.isdir(par_dir):
        os.makedirs(par_dir)
    path2save = os.path.join(par_dir, filename)
    with codecs.open(path2save, "w", encoding="utf-8") as write_file:
        json.dump(inp_dict, write_file)

def printMetric(inp_dict, title=" "):
    msg = ""
    msg += f"\n\n{title}" + "<br />"
    print(f"\n\n{title}")
    for k_method, v_method in inp_dict.items():
        msg += 10*"*****" + "<br />"
        msg += f"{k_method}" + "<br />"
        msg += "-"*len(k_method) + "<br />"
        print(10*"*****")
        print(f"{k_method}")
        print("-"*len(k_method))
        for k_metric, v_metric in v_method.items():
            for k_value, v_value in v_metric.items():
                msg += f"{k_metric} ({k_value}): {v_value}" + "<br />"
                print(f"{k_metric} ({k_value}): {v_value}")
    return msg

def printSummary(inp_dict, title="Summary"):
    print()
    print(10*"*****")
    print(f"{title}")
    print("-"*len(title))
    for k_metric, v_metric in inp_dict.items():
        for k_value, v_value in v_metric.items():
            print(f"{k_metric} ({k_value}): {v_value}")

def calc_overall_diff(util_diff):
    """Calculate mean difference across models"""
    list_methods = list(util_diff.keys())
    overall_diff_dict = {}
    for metric_k, metric_v in util_diff[list_methods[0]].items():
        overall_diff_dict[metric_k] = {}
        for avg_k, avg_v in metric_v.items():
            overall_diff_dict[metric_k][avg_k] = {}
            sum_avg = 0
            for one_method in list_methods:
                #print(metric_k, avg_k, one_method, util_diff[one_method][metric_k][avg_k])
                sum_avg += util_diff[one_method][metric_k][avg_k]
            overall_diff_dict[metric_k][avg_k] = sum_avg / len(list_methods)
    return overall_diff_dict

def calc_diff_metrics(util1, util2):
    """Calculate relative difference between two utilities"""
    util_diff = {}
    for metric_k1, metric_v1 in util1.items():
        if not metric_k1 in util2:
            continue
        util_diff[metric_k1] = {}
        for avg_k1, avg_v1 in metric_v1.items():
            if not avg_k1 in util2[metric_k1]:
                continue
            diff = abs(avg_v1 - util2[metric_k1][avg_k1]) / max(1e-9, avg_v1)
            util_diff[metric_k1][avg_k1] = diff
    return util_diff

def calc_metrics(y_pred, y_test, 
                 metrics=[("accuracy", "value"), 
                          ("precision", "macro"), 
                          ("precision", "weighted"), 
                          ("recall", "macro"), 
                          ("recall", "weighted"), 
                          ("f1", "macro"),
                          ("f1", "weighted")]):
    """Computes metrics using a list of predictions and their ground-truth labels"""
    util_collect = {}
    for method_name, ave_method in metrics:
        if not method_name in util_collect:
            util_collect[method_name] = {}
        
        if method_name.lower() in ["precision"]:
            util_collect[method_name][ave_method] = \
                precision_score(y_pred, y_test, average=ave_method, zero_division=True) * 100.
        elif method_name.lower() in ["recall"]:
            util_collect[method_name][ave_method] = \
                recall_score(y_pred, y_test, average=ave_method, zero_division=True) * 100.
        elif method_name.lower() in ["f1", "f-1"]:
            util_collect[method_name][ave_method] = \
                f1_score(y_pred, y_test, average=ave_method, zero_division=True) * 100.
        elif method_name.lower() in ["accuracy"]:
            util_collect[method_name][ave_method] = \
                accuracy_score(y_pred, y_test) * 100.
    return util_collect


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
        '-o', dest='outfile_prefix', required=True,
        help='The prefix of the output paths (data json and csv), relative to the QUIPP-pipeline root directory')

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
    synth_method = synth_params["synth-method"]
    path_original_meta = os.path.abspath(dataset) + '.json'
    path_released_ds = args.outfile_prefix
    if synth_method == 'sgf':
        path_original_ds = os.path.join(path_released_ds, os.path.basename(dataset) + "_numcat.csv")
    else:
        path_original_ds = os.path.abspath(dataset) + '.csv'

    # read parameters from .json
    # parameters = synth_params["parameters"]
    sklearn_utility_parameters = synth_params["parameters_sklearn_utility"]

    input_columns = sklearn_utility_parameters["input_columns"]
    label_column = sklearn_utility_parameters["label_column"]
    test_train_ratio = sklearn_utility_parameters["test_train_ratio"]
    output_file_json = path_released_ds + "/sklearn_classifiers.json"
    num_leaked_rows = sklearn_utility_parameters["num_leaked_rows"]
    seed = synth_params['parameters']['random_state']

    print("\n=================================================")
    print("[WARNING] the classifiers are hardcoded in main()")
    print("=================================================\n")
    # List of classifiers and their arguments
    classifiers = {
        LogisticRegression:  {"mode": "range",
                             "params_main": {"max_iter": 5000},
                             "params_range": {"classifier__max_iter": [10,50,100,150,180, 200, 250, 300]}
                             },
        KNeighborsClassifier: {"mode": "main",
                              "params_main": {"n_neighbors": 3},
                              "params_range": {"classifier__n_neighbors": [3, 4, 5]}
                              },
        SVC: {"mode": "range",
             "params_main": {"kernel": "linear", "C": 0.025},
             "params_range": {'classifier__C': [0.025, 0.05, 0.1, 0.5, 1], "classifier__kernel": ("linear", "rbf")}
             },
        # SVC: {"gamma": 2, "C": 1},
        
        #GaussianProcessClassifier: {"mode": "main", 
        #                            "params_main": {"kernel": 1.0 * RBF(1.0)},
        #                            "params_range": {}
        #                            },

        RandomForestClassifier: {"mode": "main", 
                                 "params_main": {"max_depth": 5, "n_estimators": 10, "max_features": 1, "random_state": 123},
                                 "params_range": {}
                                 },

        # DecisionTreeClassifier: {"max_depth": 5},
        # MLPClassifier: {"alpha": 1, "max_iter": 5000},
        # AdaBoostClassifier: {},
        #GaussianNB: {},
        #QuadraticDiscriminantAnalysis: {}
    }


    utility_measure_sklearn_classifiers(synth_method,
                                        path_original_ds,
                                        path_original_meta,
                                        path_released_ds,
                                        input_columns,
                                        label_column,
                                        test_train_ratio,
                                        classifiers,
                                        output_file_json,
                                        num_leaked_rows)


if __name__ == '__main__':
    main()
