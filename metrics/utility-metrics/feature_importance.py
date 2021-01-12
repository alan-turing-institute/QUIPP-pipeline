"""
Code to calculate feature importance utility metrics
"""

import featuretools as ft
import featuretools.variable_types as vtypes
import json
import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, "utilities"))
from utils import handle_cmdline_args, extract_parameters, find_column_types


def featuretools_importances(df, data_meta, utility_params_ft):
    data_json_type_to_vtype = {'Categorical': vtypes.Categorical,
                               'ContinuousNumerical': vtypes.Numeric,
                               'DateTime': vtypes.Datetime,
                               'DiscreteNumerical': vtypes.Ordinal,
                               'Ordinal': vtypes.Ordinal,
                               'String': vtypes.Categorical}

    entity_id = 'my_entity_id'
    
    variable_types = { m['name'] : data_json_type_to_vtype[m['type']]
                       for m in data_meta['columns'] }
    
    es = ft.EntitySet('myEntitySet')

    es = es.entity_from_dataframe(entity_id=entity_id,
                                  dataframe=df,
                                  index=utility_params_ft['entity_index'],
                                  time_index=utility_params_ft.get('time_index'),
                                  secondary_time_index=utility_params_ft.get('secondary_time_index'),
                                  variable_types=variable_types)

    for ne in utility_params_ft.get('normalized_entities'):
        es.normalize_entity(base_entity_id=entity_id, **ne)

    cutoff_times = (es[entity_id]
                    .df[[utility_params_ft['entity_index'],
                         utility_params_ft['time_index'],
                         utility_params_ft['label_column']]]
                    .sort_values(by=utility_params_ft['time_index']))

    fm, features = ft.dfs(entityset=es,
                          target_entity=entity_id,
                          agg_primitives=['count', 'percent_true'],
                          trans_primitives=['is_weekend', 'weekday', 'day', 'month', 'year'],
                          max_depth=3,
                          approximate='6h',
                          cutoff_time=cutoff_times)

    ## Cannot use strings ('objects') as features
    ## See https://stackoverflow.com/questions/40913104/how-to-use-randomforestclassifier-with-string-data#40934357
    ## TODO: bag of words
    Y = fm.pop(utility_params_ft['label_column'])
    X = fm[fm.dtypes.index[[x is not np.dtype('object') for x in fm.dtypes]]]

    clf = RandomForestClassifier(n_estimators=150)
    clf.fit(X, Y)

    feature_imps = [(imp, X.columns[i]) for i, imp in enumerate(clf.feature_importances_)]
    feature_imps.sort()
    feature_imps.reverse()

    return feature_imps


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
        pass
        # calculate metric...
        ## TODO:
        ##  1. call featuretools_importances on original data
        ##  2. call featuretools_importances on synthetic data
        ##  3. fix datetimes in synthetic data (2. will fail)

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

    utility_params_ft = synth_params['utility_parameters_feature_importance']
        
    # if the whole .json is not enabled or if the
    # feature importance utility metrics are not enabled, stop here
    if not (synth_params["enabled"] and utility_params_ft['enabled']):
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
