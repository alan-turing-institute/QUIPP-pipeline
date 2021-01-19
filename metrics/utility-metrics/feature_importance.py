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
        "DiscreteNumerical": vtypes.Ordinal,
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
                    utility_params_ft["entity_index"],
                    utility_params_ft["time_index"],
                    utility_params_ft["label_column"],
                ]
            ]
                .sort_values(by=utility_params_ft["time_index"])
        )
    else:
        cutoff_times = None

    fm, features = ft.dfs(
        entityset=es,
        target_entity=entity_id,
        agg_primitives=["count", "percent_true"],
        trans_primitives=["is_weekend", "weekday", "day", "month", "year"],
        max_depth=3,
        approximate="6h",
        cutoff_time=cutoff_times,
    )

    # drop null/nan values to allow sklearn to fit the RF model
    fm = fm.dropna()

    Y = fm.pop(utility_params_ft["label_column"])

    # create dummies for string categorical variables
    # drops last dummy column for each variable
    for col in fm.dtypes.index[[x is np.dtype("object") for x in fm.dtypes]]:
        one_hot = pd.get_dummies(fm[col]).iloc[:,0:-1]
        fm = fm.drop(col, axis=1)
        fm = fm.join(one_hot, rsuffix="_" + col)

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
    print('AUC score of {:.3f}'.format(roc_auc_score(y_test, probs[:, 1])))

    feature_imps = [
        (imp, fm.columns[i]) for i, imp in enumerate(clf.feature_importances_)
    ]
    feature_imps.sort()
    feature_imps.reverse()

    return feature_imps


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
    if path_released_ds is not None:
        rlsd_df = pd.read_csv(os.path.join(path_released_ds, "synthetic_data_1.csv"))

    # store metrics in dictionary
    utility_collector = {}

    with warnings.catch_warnings(record=True) as warns:
        print("[INFO] Compute feature importance for original dataset")
        orig_feature_importances = featuretools_importances(orig_df, orig_metadata, utility_params, random_seed)
        rank_orig_features = [i[1] for i in orig_feature_importances]
        score_orig_features = [i[0] for i in orig_feature_importances]
        
        if path_released_ds is not None:
            print("[INFO] Compute feature importance for synthetic dataset")
            rlsd_feature_importances = featuretools_importances(rlsd_df, orig_metadata, utility_params, random_seed)
            rank_rlsd_features = [i[1] for i in rlsd_feature_importances]
            score_rlsd_features = [i[0] for i in rlsd_feature_importances]
            utility_collector = compare_features(rank_orig_features, rank_rlsd_features,
                                                 score_orig_features, score_rlsd_features,
                                                 utility_collector, percentage_threshold)
            utility_collector["orig_feature_importances"] = orig_feature_importances
            utility_collector["rlsd_feature_importances"] = rlsd_feature_importances
        else:
            final_collector = []

            for rs in random_seeds_rf:
                orig_feature_importances_2 = featuretools_importances(orig_df, orig_metadata, utility_params, rs)
                rank_orig_features_2 = [i[1] for i in orig_feature_importances_2]
                score_orig_features_2 = [i[0] for i in orig_feature_importances_2]
                utility_collector = compare_features(rank_orig_features, rank_orig_features_2,
                                                     score_orig_features, score_orig_features_2,
                                                     utility_collector, percentage_threshold)
                utility_collector[f"orig_feature_importances_{random_seed}"] = orig_feature_importances
                utility_collector[f"orig_feature_importances_{rs}"] = orig_feature_importances_2
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
                     score_orig_features: Union[None, list]=None, 
                     score_rlsd_features: Union[None, list]=None, 
                     utility_collector: dict={}, 
                     percentage_threshold: Union[float, None]=None):
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
    
    utility_collector["rbo_0.6"] = RankingSimilarity(rank_orig_features[:target_index], 
                                                     rank_rlsd_features[:target_index]).rbo(p=0.6)
    
    utility_collector["rbo_0.8"] = RankingSimilarity(rank_orig_features[:target_index], 
                                                     rank_rlsd_features[:target_index]).rbo(p=0.8)

    # Rank-Biased Overlap (RBO), extrapolated version
    utility_collector["rbo_ext_0.6"] = RankingSimilarity(rank_orig_features[:target_index], 
                                                         rank_rlsd_features[:target_index]).rbo_ext(p=0.6)

    utility_collector["rbo_ext_0.8"] = RankingSimilarity(rank_orig_features[:target_index], 
                                                         rank_rlsd_features[:target_index]).rbo_ext(p=0.8)
    
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
        utility_collector["l2_norm"] = np.sqrt(np.sum(diff_orig_rlsd_scores**2))

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
        random_seed,
    )


if __name__ == "__main__":
    main()
