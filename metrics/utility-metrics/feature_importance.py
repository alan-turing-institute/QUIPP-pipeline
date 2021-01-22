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
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder
from typing import Union

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, "utilities"))
from utils import handle_cmdline_args, extract_parameters, find_column_types

sys.path.append(os.path.join(os.path.dirname(__file__)))
from rbo import RankingSimilarity


def featuretools_importances(df, data_meta, utility_params_ft, rs):
    data_json_type_to_vtype = {
        "Categorical": vtypes.Categorical,
        "ContinuousNumerical": vtypes.Numeric,
        "DateTime": vtypes.Datetime,
        "DiscreteNumerical": vtypes.Numeric,
        "Ordinal": vtypes.Ordinal,
        "String": vtypes.Categorical,
    }

    entity_id = "my_entity_id"

    variable_types = {
        m["name"]: data_json_type_to_vtype[m["type"]] for m in data_meta["columns"]
    }

    es = ft.EntitySet("myEntitySet")

    # if there is no id/index column in the dataframe, create one
    # for use with featuretools
    index = utility_params_ft.get("entity_index")
    if index is None:
        df['index_col'] = df.index
        index = "index_col"

    # drop nan
    df = df.dropna(axis=0, how='any')

    es = es.entity_from_dataframe(
        entity_id=entity_id,
        dataframe=df,
        index=index,
        time_index=utility_params_ft.get("time_index"),
        secondary_time_index=utility_params_ft.get("secondary_time_index"),
        variable_types=variable_types
    )

    for ne in utility_params_ft.get("normalized_entities"):
        es.normalize_entity(base_entity_id=entity_id, **ne)

    if utility_params_ft.get("time_index") is not None:
        cutoff_times = (
            es[entity_id]
                .df[
                [
                    index,
                    utility_params_ft["time_index"],
                    utility_params_ft["label_column"],
                ]
            ]
                .sort_values(by=utility_params_ft["time_index"])
        )
    else:
        cutoff_times = None

    max_depth_param = utility_params_ft.get("max_depth")
    if max_depth_param is None:
        max_depth = 3
    else:
        max_depth = max_depth_param

    fm, features = ft.dfs(
        entityset=es,
        target_entity=entity_id,
        agg_primitives=utility_params_ft["aggPrimitives"],
        trans_primitives=utility_params_ft["tranPrimitives"],
        max_depth=max_depth,
        approximate="6h",
        cutoff_time=cutoff_times,
    )

    fm = fm.replace([np.inf, -np.inf], np.nan)

    # drop null/nan values to allow sklearn to fit the RF model
    if utility_params_ft.get("drop_na") is not None:
        if utility_params_ft["drop_na"] == "columns":
            fm = fm.dropna(axis=1, how='any')
        elif utility_params_ft["drop_na"] == "rows":
            fm = fm.dropna()
        elif utility_params_ft["drop_na"] == "all":
            fm = fm.dropna()
            fm = fm.dropna(axis=1, how='any')

    Y = fm.pop(utility_params_ft["label_column"])

    ## create dummies or numerical labels for string categorical variables    
    for col in fm.dtypes.index[[x is np.dtype("object") for x in fm.dtypes]]:
        if utility_params_ft["categorical_enconding"] == "dummies":
            one_hot = pd.get_dummies(fm[col], prefix=col, prefix_sep="_")#.iloc[:, 0:-1]
            fm = fm.drop(col, axis=1)
            fm = fm.join(one_hot)
        else:
            labelencoder = LabelEncoder()
            fm[col] = labelencoder.fit_transform(fm[col])

    # drop columns that the user wants to exclude
    ef = utility_params_ft.get("features_to_exclude")
    if ef is not None:
        fm = fm.drop(ef, axis=1)

    # split data into train and test sets
    fm_train, fm_test, y_train, y_test = train_test_split(fm, Y, test_size=0.30, shuffle=False)

    # train Random Forest model
    clf = RandomForestClassifier(n_estimators=150, random_state=rs)
    clf.fit(fm_train, y_train)

    # predict test labels and calculate AUC
    probs = clf.predict_proba(fm_test)
    auc = roc_auc_score(y_test, probs[:, 1])
    print('AUC score of {:.3f}'.format(auc))

    # get built-in RF feature importances
    feature_imps_builtin = [
        (imp, fm.columns[i]) for i, imp in enumerate(clf.feature_importances_)
    ]
    feature_imps_builtin.sort()
    feature_imps_builtin.reverse()

    # get permutation feature importances from RF
    feature_imps_permutation = permutation_importance(clf, fm_test, y_test)
    sorted_idx = feature_imps_permutation.importances_mean.argsort()[::-1]
    feature_imps_permutation = list(zip(feature_imps_permutation.importances_mean[sorted_idx],
                                        fm.columns[sorted_idx]))

    return auc, feature_imps_builtin, feature_imps_permutation


def feature_importance_metrics(
        path_original_ds,
        path_original_meta,
        path_released_ds,
        utility_params,
        output_file_json,
        percentage_threshold=None,
        random_seed=1234,
        random_seeds_rf=[124331, 111233, 554365, 976873, 123874]
):
    """
    Calculates feature importance ranking differences between the original
    and released datasets, using a random forest classification model
    and the RBO ranking comparison metric.
    Saves the results into a .json file. These can be compared to
    estimate the utility of the released dataset.

    If path_released_ds is None, it calculates differences between the
    rankings produced when changing the the random forest seed many
    times, while always using the original dataset for training.

    Parameters
    ----------
    path_original_ds : string
        Path to the original dataset.
    path_original_meta : string
        Path to the original metadata.
    path_released_ds : string
        Path to the released dataset. If None, then calculate difference between
        rankings produced with different RF seeds on the original dataset.
    utility_params: dict
        Parameters for feature importance utility metrics read from inputs json file.
    output_file_json : string
        Path to the output json file that will be generated.
    percentage_threshold : Union[float, None], optional
        if None, all features will be used in computing Rank-biased Overlap (RBO)
        otherwise, if percentage_threshold is a float between 0 and 1 
            1. the cumulative sum of the ranking scores is calculated (original dataset)
            2. select only those features that their scores contribute to the specified percentage_threshold
            3. compute RBO for the selected features
    random_seed : integer
        Random seed for numpy. Defaults to 1234
    random_seeds_rf : list
        Random seeds list for random forest training. Only used when path_released_ds is None.
    """

    print("[INFO] Calculating feature importance utility metricsß")

    # set random seed
    np.random.seed(random_seed)

    # read metadata in JSON format
    with open(path_original_meta) as orig_metadata_json:
        orig_metadata = json.load(orig_metadata_json)

    # read original and released/synthetic datasets,
    # only the first synthetic data set (synthetic_data_1.csv)
    # is used for utility evaluation
    orig_df = pd.read_csv(path_original_ds)
    if path_released_ds is not None:
        rlsd_df = pd.read_csv(os.path.join(path_released_ds, "synthetic_data_1.csv"))

    # store metrics in dictionary
    utility_collector_builtin = {}
    utility_collector_permutation = {}

    with warnings.catch_warnings(record=True) as warns:
        print("Computing feature importance for original dataset")
        orig_feature_importances_builtin, orig_feature_importances_permutation = \
            featuretools_importances(orig_df, orig_metadata, utility_params, random_seed)
        rank_orig_features_builtin = [i[1] for i in orig_feature_importances_builtin]
        score_orig_features_builtin = [i[0] for i in orig_feature_importances_builtin]
        rank_orig_features_permutation = [i[1] for i in orig_feature_importances_permutation]
        score_orig_features_permutation = [i[0] for i in orig_feature_importances_permutation]

        if path_released_ds is not None:
            print("Computing feature importance for synthetic dataset")
            auc, rlsd_feature_importances_builtin, rlsd_feature_importances_permutation = \
                featuretools_importances(rlsd_df, orig_metadata, utility_params, random_seed)
            rank_rlsd_features_builtin = [i[1] for i in rlsd_feature_importances_builtin]
            score_rlsd_features_builtin = [i[0] for i in rlsd_feature_importances_builtin]
            rank_rlsd_features_permutation = [i[1] for i in rlsd_feature_importances_permutation]
            score_rlsd_features_permutation = [i[0] for i in rlsd_feature_importances_permutation]

            utility_collector_builtin = compare_features(rank_orig_features_builtin, rank_rlsd_features_builtin,
                                                         score_orig_features_builtin, score_rlsd_features_builtin,
                                                         utility_collector_builtin, percentage_threshold)
            utility_collector_builtin["orig_feature_importances"] = orig_feature_importances_builtin
            utility_collector_builtin["rlsd_feature_importances"] = rlsd_feature_importances_builtin
            utility_collector_builtin["auc"] = auc

            utility_collector_permutation = compare_features(rank_orig_features_permutation,
                                                             rank_rlsd_features_permutation,
                                                             score_orig_features_permutation,
                                                             score_rlsd_features_permutation,
                                                             utility_collector_permutation, percentage_threshold)
            utility_collector_permutation["orig_feature_importances"] = orig_feature_importances_permutation
            utility_collector_permutation["rlsd_feature_importances"] = rlsd_feature_importances_permutation
            utility_collector_permutation["auc"] = auc

            utility_collector = {"builtin": utility_collector_builtin, "permutation": utility_collector_permutation}

        else:
            print("Computing feature importance for original dataset with different seeds in RF")
            final_collector = []
            for rs in random_seeds_rf:
                auc, orig_feature_importances_builtin_2, orig_feature_importances_permutation_2 = \
                    featuretools_importances(orig_df, orig_metadata, utility_params, rs)
                rank_orig_features_builtin_2 = [i[1] for i in orig_feature_importances_builtin_2]
                score_orig_features_builtin_2 = [i[0] for i in orig_feature_importances_builtin_2]

                utility_collector_builtin = compare_features(rank_orig_features_builtin, rank_orig_features_builtin_2,
                                                             score_orig_features_builtin, score_orig_features_builtin_2,
                                                             utility_collector_builtin, percentage_threshold)
                utility_collector_builtin[f"orig_feature_importances_{random_seed}"] = orig_feature_importances_builtin
                utility_collector_builtin[f"orig_feature_importances_{rs}"] = orig_feature_importances_builtin_2
                utility_collector_builtin[f"auc_{rs}"] = auc

                rank_orig_features_permutation_2 = [i[1] for i in orig_feature_importances_permutation_2]
                score_orig_features_permutation_2 = [i[0] for i in orig_feature_importances_permutation_2]
                utility_collector_permutation = compare_features(rank_orig_features_permutation,
                                                                 rank_orig_features_permutation_2,
                                                                 score_orig_features_permutation,
                                                                 score_orig_features_permutation_2,
                                                                 utility_collector_permutation, percentage_threshold)
                utility_collector_permutation[
                    f"orig_feature_importances_{random_seed}"] = orig_feature_importances_permutation
                utility_collector_permutation[f"orig_feature_importances_{rs}"] = orig_feature_importances_permutation_2
                utility_collector_permutation[f"auc_{rs}"] = auc

                utility_collector = {"builtin": utility_collector_builtin, "permutation": utility_collector_permutation}

                final_collector.append(utility_collector)
                utility_collector = {}

        ##  3. fix datetimes in synthetic data (2. will fail)

    # print warnings
    if len(warns) > 0:
        print("WARNINGS:")
        for iw in warns:
            print(iw.message)

    # save as .json
    if path_released_ds is not None:
        with open(output_file_json, "w") as out_fio:
            json.dump(utility_collector, out_fio, indent=4)
    else:
        with open(output_file_json, "w") as out_fio:
            json.dump(final_collector, out_fio, indent=4)


def compare_features(rank_orig_features: list, rank_rlsd_features: list,
                     score_orig_features: Union[None, list] = None,
                     score_rlsd_features: Union[None, list] = None,
                     utility_collector: dict = {},
                     percentage_threshold: Union[float, None] = None):
    """Compare ranked features using different methods including
        Ranked-biased Overlap (RBO), extrapolated version of RBO, 
        L2 norm and KL divergence

    Parameters
    ----------
    rank_orig_features : list
        ranked features from the original dataset
    rank_rlsd_features : list
        ranked features from the synthetic/released dataset
    score_orig_features : Union[None, list], optional
        scores of the ranked features from the original dataset, by default None
    score_rlsd_features : Union[None, list], optional
        scores of the ranked features from the synthetic/released dataset, by default None
    utility_collector : dict, optional
        a dictionary to collect utilities, by default {}
    percentage_threshold : Union[float, None], optional
        if None, all features will be used in computing Rank-biased Overlap (RBO)
        otherwise, if percentage_threshold is a float between 0 and 1 
            1. the cumulative sum of the ranking scores is calculated (original dataset)
            2. select only those features that their scores contribute to the specified percentage_threshold
            3. compute RBO for the selected features
    """

    if percentage_threshold != None and (0 < percentage_threshold < 1) and score_orig_features != None:
        target_index = np.argmax(np.cumsum(score_orig_features) > percentage_threshold)
    else:
        target_index = len(rank_orig_features)

    # Rank-Biased Overlap (RBO)

    orig_rlsd_sim = RankingSimilarity(rank_orig_features[:target_index],
                                      rank_rlsd_features[:target_index])

    utility_collector["rbo_0.6"] = orig_rlsd_sim.rbo(p=0.6)
    utility_collector["rbo_0.8"] = orig_rlsd_sim.rbo(p=0.8)

    # extrapolated version
    utility_collector["rbo_ext_0.6"] = orig_rlsd_sim.rbo_ext(p=0.6)
    utility_collector["rbo_ext_0.8"] = orig_rlsd_sim.rbo_ext(p=0.8)

    # original against one random permutation
    orig_rand_sim = RankingSimilarity(rank_orig_features[:target_index],
                                      np.random.permutation(rank_orig_features[:target_index]))

    utility_collector["rbo_rand_0.6"] = orig_rand_sim.rbo(p=0.6)
    utility_collector["rbo_rand_0.8"] = orig_rand_sim.rbo(p=0.8)

    # original lower bound
    orig_lower_sim = RankingSimilarity(rank_orig_features[:target_index],
                                       list(reversed(rank_orig_features[:target_index])))

    utility_collector["rbo_lower_0.6"] = orig_lower_sim.rbo(p=0.6)
    utility_collector["rbo_lower_0.8"] = orig_lower_sim.rbo(p=0.8)

    if score_orig_features != None and score_rlsd_features != None:
        # L2 norm
        tmp_orig_df = pd.DataFrame(score_orig_features, columns=["score_orig_features"])
        tmp_orig_df["rank_orig_features"] = rank_orig_features

        tmp_rlsd_df = pd.DataFrame(score_rlsd_features, columns=["score_rlsd_features"])
        tmp_rlsd_df["rank_rlsd_features"] = rank_rlsd_features

        orig_rlsd_df = pd.merge(tmp_orig_df, tmp_rlsd_df,
                                left_on="rank_orig_features",
                                right_on="rank_rlsd_features")

        diff_orig_rlsd_scores = (orig_rlsd_df["score_orig_features"] - orig_rlsd_df["score_rlsd_features"]).to_numpy()
        utility_collector["l2_norm"] = np.sqrt(np.sum(diff_orig_rlsd_scores ** 2))

        # KL divergence
        orig_sf = orig_rlsd_df["score_orig_features"].to_numpy() + 1e-20
        rlsd_sf = orig_rlsd_df["score_rlsd_features"].to_numpy() + 1e-20
        utility_collector["kl_orig_rlsd"] = entropy(orig_sf, rlsd_sf)
    return utility_collector


def main():
    # process command line arguments
    args = handle_cmdline_args()

    # read run input parameters file
    with open(args.infile) as f:
        synth_params = json.load(f)

    utility_params_ft = synth_params["utility_parameters_feature_importance"]

    # if the whole .json is not enabled or if the
    # feature importance utility metrics are not enabled, stop here
    if not (synth_params["enabled"] and utility_params_ft["enabled"]):
        return

    # extract paths and other parameters from args
    (
        synth_method,
        path_original_ds,
        path_original_meta,
        path_released_ds,
        random_seed,
    ) = extract_parameters(args, synth_params)

    # create output .json full path
    output_file_json = os.path.join(path_released_ds, "utility_feature_importance.json")

    # calculate and save feature importance metrics
    feature_importance_metrics(
        path_original_ds,
        path_original_meta,
        path_released_ds,
        utility_params_ft,
        output_file_json,
        None,
        random_seed,
    )


if __name__ == "__main__":
    main()
