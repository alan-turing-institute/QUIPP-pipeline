"""
Code to calculate feature importance utility metrics
"""

import featuretools as ft
import featuretools.variable_types as vtypes
import json
import os
import shap
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.metrics.pairwise import cosine_similarity
from typing import Union
from dython.nominal import associations
from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable, constants, value
import pulp
pulp.LpSolverDefault.msg = 1

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, "utilities"))
from utils import handle_cmdline_args, extract_parameters, find_column_types

sys.path.append(os.path.join(os.path.dirname(__file__)))
from rbo import RankingSimilarity


def featuretools_importances(df, data_meta, utility_params_ft, rs):

    if utility_params_ft.get("skip_feature_engineering"):

        fm = df.copy(deep=False)

    else:

        # This should be run only for the household poverty dataset
        # It finds all households which have a head of household
        # This is later used to filter out the households without a head
        if utility_params_ft.get("filter_hh"):
            valid_hh = df.loc[df['parentesco'] == 0, ['idhogar', 'Target']].copy()

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

        if not utility_params_ft.get("target_entity"):
            te = entity_id
        else:
            te = utility_params_ft.get("target_entity")

        fm, features = ft.dfs(
            entityset=es,
            target_entity=te,
            agg_primitives=utility_params_ft.get("aggPrimitives"),
            trans_primitives=utility_params_ft.get("tranPrimitives"),
            max_depth=max_depth,
            approximate="6h",
            cutoff_time=cutoff_times,
        )

    #  drop any columns derived from the label column
    derived_cols = []
    for col in fm:
        if col == utility_params_ft["label_column"]:
            pass
        else:
            if utility_params_ft["label_column"] in col:
                derived_cols.append(col)
    fm = fm[[x for x in fm if x not in derived_cols]]

    # replace inf with nan
    fm = fm.replace([np.inf, -np.inf], np.nan)

    # drop null/nan values to allow sklearn to fit the RF model
    if utility_params_ft.get("drop_full_na_columns"):
        fm_temp = fm.dropna(axis=1, thresh=int(fm.shape[0] * utility_params_ft.get("na_thresh")))
        dropped_columns = list(set(fm.columns) - set(fm_temp.columns))
        fm.dropna(axis=1, thresh=int(fm.shape[0] * utility_params_ft.get("na_thresh")), inplace=True)
        print(f"Dropped {len(dropped_columns)} columns: {dropped_columns}")
    else:
        dropped_columns = []

    if utility_params_ft.get("drop_na"):
        fm = fm.dropna()

    # replace +,-,*,/,< with words to avoid errors later on when using pulp for optimisation
    fm.columns = [col.replace("<", "LESSTHAN") for col in
                  [col.replace("/", "DIV") for col in
                  [col.replace("*", "MULT") for col in
                   [col.replace("-", "SUB") for col in
                    [col.replace("+", "ADD") for col in
                     fm.columns]]]]]

    for col in fm:
        if col == utility_params_ft["label_column"]:
            pass
        else:
            if utility_params_ft["label_column"] in col:
                derived_cols.append(col)

    # This should only run for the household poverty dataset to filter
    # out invalid households
    if utility_params_ft.get("filter_hh"):
        fm.reset_index(inplace=True)
        fm = fm[fm['idhogar'].isin(list(valid_hh['idhogar']))]

    Y = fm.pop(utility_params_ft["label_column"])

    # create dummies or numerical labels for string categorical variables
    for col in fm.dtypes.index[[x is np.dtype("object") for x in fm.dtypes]]:
        if utility_params_ft.get("categorical_enconding") == "dummies":
            one_hot = pd.get_dummies(fm[col], prefix=col, prefix_sep="_")#.iloc[:, 0:-1]
            fm = fm.drop(col, axis=1)
            fm = fm.join(one_hot)
        elif utility_params_ft.get("categorical_enconding") == "labels":
            labelencoder = LabelEncoder()
            fm[col] = labelencoder.fit_transform(fm[col])
        else:
            pass

    # drop columns that the user wants to exclude
    ef = utility_params_ft.get("features_to_exclude")
    if ef is not None:
        fm = fm.drop(ef, axis=1)

    # split data into train and test sets
    fm_train, fm_test, y_train, y_test = train_test_split(fm, Y, test_size=0.30, random_state=rs)

    # train Random Forest model
    ne = utility_params_ft.get("rf_n_estimators")
    if ne is None:
        ne = 150
    clf = RandomForestClassifier(n_estimators=ne, max_depth=utility_params_ft.get("rf_max_depth"), random_state=rs)
    clf.fit(fm_train, y_train)

    # predict test labels and calculate AUC

    probs = clf.predict_proba(fm_test)
    y_pred = clf.predict(fm_test)
    if len(Y.unique()) > 2:
        multiclass = True
        try:
            auc = roc_auc_score(y_test, probs, multi_class="ovo")
            f1 = f1_score(y_test, y_pred, average='weighted')
        except ValueError:
            auc = float('NaN')
            f1 = float('NaN')
            print(f"AUC and F1 set to NaN because number of classes in the training sample is "
                  f"different to the number of classes in the test sample ({probs.shape[1]} vs {len(set(y_test))}). "
                  f"This is usually caused when the training/test samples are imbalanced or very small.")
    else:
        multiclass = False
        try:
            auc = roc_auc_score(y_test, probs[:, 1])
            f1 = f1_score(y_test, y_pred, average='weighted')
        except:
            auc = float('NaN')
            f1 = float('NaN')
            print("AUC and F1 set to NaN because only one class is present "
                  "in the probability estimates array (generated by predict_proba()). "
                  "This is usually caused when the training sample is imbalanced or very small.")

    print('AUC score of {:.3f} and weighted F1 score of {:.3f}'.format(auc, f1))

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

    # get Shapley values
    if utility_params_ft.get("compute_shapley"):
        explainer = shap.TreeExplainer(model=clf)
        shap_values = explainer.shap_values(fm_test)
        # version that uses the interventional perturbation option (takes into account a background dataset
        # fm_train) - throws errors in some cases which can be suppressed by setting check_additivity=False
        # in explainer.shap_values(). It is also slower.
        # explainer = shap.TreeExplainer(model=clf, data=fm_train, feature_perturbation='interventional')
        # shap_values = explainer.shap_values(fm_test, check_additivity=False)
        vals = np.abs(shap_values).mean(0)
        try:
            feature_imps_shapley = pd.DataFrame(list(zip(fm_test.columns, sum(vals))),
                                                columns=['col_name', 'feature_importance_vals'])
        except TypeError:
            # Numpy type error happens when all shapley values are zeros. Artificially creating an all zeros
            # feature importance data frame
            feature_imps_shapley = pd.DataFrame(list(zip(fm_test.columns, np.zeros(fm_test.columns.shape))),
                                                columns=['col_name', 'feature_importance_vals'])
        feature_imps_shapley.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
        feature_imps_shapley = [(row['feature_importance_vals'], row['col_name'])
                                for index, row in feature_imps_shapley.iterrows()]
    else:
        feature_imps_shapley = []
        
    return auc, f1, feature_imps_builtin, feature_imps_permutation, \
           feature_imps_shapley, clf, fm_test, y_test, dropped_columns, multiclass, fm_train

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
    and released datasets, using a random forest classification model, various
    feature importance measures and various feature rank/score comparison measures.
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

    print("[INFO] Calculating feature importance utility metrics")

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
    utility_collector_shapley = {}

    with warnings.catch_warnings(record=True) as warns:
        print("Computing feature importance for original dataset")
        auc_orig, f1_orig, orig_feature_importances_builtin, orig_feature_importances_permutation, \
        orig_feature_importances_shapley, _, X_test_orig, y_test_orig, dropped_cols_orig, multiclass_orig, \
        X_train_orig = featuretools_importances(orig_df, orig_metadata, utility_params, random_seed)
        rank_orig_features_builtin = [i[1] for i in orig_feature_importances_builtin]
        score_orig_features_builtin = [i[0] for i in orig_feature_importances_builtin]
        rank_orig_features_permutation = [i[1] for i in orig_feature_importances_permutation]
        score_orig_features_permutation = [i[0] for i in orig_feature_importances_permutation]
        rank_orig_features_shapley = [i[1] for i in orig_feature_importances_shapley]
        score_orig_features_shapley = [i[0] for i in orig_feature_importances_shapley]

        if path_released_ds is not None:
            print("Computing feature importance for synthetic dataset")
            auc_rlsd, f1_rlsd, rlsd_feature_importances_builtin, rlsd_feature_importances_permutation, \
            rlsd_feature_importances_shapley, clf_rlsd, _, _, dropped_cols_rlsd, multiclass_rlsd, _ = \
                featuretools_importances(rlsd_df, orig_metadata, utility_params, random_seed)

            # predict test labels and calculate cross AUC - trained on rlsd, test on original
            # first step is to drop from the orig test set the columns that were dropped in the rlsd (if any)
            if len(dropped_cols_orig) > 0:
                raise ValueError("Columns were dropped from the original dataset")
            if len(dropped_cols_rlsd) > 0:
                X_test_orig.drop(dropped_cols_rlsd, axis=1, inplace=True)
            probs = clf_rlsd.predict_proba(X_test_orig)
            y_pred = clf_rlsd.predict(X_test_orig)
            # if this is a multiclass problem, different invocation of the AUC metric is used
            if multiclass_orig:
                try:
                    auc_cross = roc_auc_score(y_test_orig, probs, multi_class="ovo")
                    f1_cross = f1_score(y_test_orig, y_pred, average='weighted')
                except ValueError:
                    auc_cross = float('NaN')
                    f1_cross = float('NaN')
                    print(f"Cross-AUC set to NaN because number of classes in the model trained on the "
                          f"synthetic data is different to the number of classes in the test "
                          f"sample coming from the original data ({probs.shape[1]} vs {len(set(y_test_orig))}). "
                          f"This is usually caused when the training/test samples are imbalanced or very small.")
            else:
                try:
                    auc_cross = roc_auc_score(y_test_orig, probs[:, 1])
                    f1_cross = f1_score(y_test_orig, y_pred, average='weighted')
                except:
                    auc_cross = float('NaN')
                    f1_cross = float('NaN')
                    print("Cross-AUC and Cross-F1 set to NaN because only one class is present "
                          "in the probability estimates array (generated by predict_proba()). "
                          "This is usually caused when the training/test samples are imbalanced or very small.")

            print('Cross-AUC score of {:.3f} and weighted Cross-F1 of {:.3f}'.format(auc_cross, f1_cross))

            # Correlation matrix needed for correlated rank similarity
            categorical_columns = [c["name"] for c in orig_metadata["columns"]
                                   if (c["type"] in ["Categorical", "Ordinal"]
                                   and c["name"] in X_train_orig.columns)]
            categorical_columns.extend([c for c in X_train_orig.columns if "MODE" in c])
            correlation_matrix = associations(X_train_orig, nominal_columns=categorical_columns,
                                              plot=False)["corr"].abs()

            rank_rlsd_features_builtin = [i[1] for i in rlsd_feature_importances_builtin]
            score_rlsd_features_builtin = [i[0] for i in rlsd_feature_importances_builtin]
            rank_rlsd_features_permutation = [i[1] for i in rlsd_feature_importances_permutation]
            score_rlsd_features_permutation = [i[0] for i in rlsd_feature_importances_permutation]
            rank_rlsd_features_shapley = [i[1] for i in rlsd_feature_importances_shapley]
            score_rlsd_features_shapley = [i[0] for i in rlsd_feature_importances_shapley]

            utility_collector_builtin = compare_features(rank_orig_features_builtin,
                                                         rank_rlsd_features_builtin,
                                                         correlation_matrix,
                                                         score_orig_features_builtin,
                                                         score_rlsd_features_builtin,
                                                         utility_collector_builtin,
                                                         percentage_threshold)

            utility_collector_builtin["orig_feature_importances"] = orig_feature_importances_builtin
            utility_collector_builtin["rlsd_feature_importances"] = rlsd_feature_importances_builtin
            utility_collector_builtin["auc_orig"] = auc_orig
            utility_collector_builtin["auc_rlsd"] = auc_rlsd
            utility_collector_builtin["auc_cross"] = auc_cross
            utility_collector_builtin["f1_orig"] = f1_orig
            utility_collector_builtin["f1_rlsd"] = f1_rlsd
            utility_collector_builtin["f1_cross"] = f1_cross

            utility_collector_permutation = compare_features(rank_orig_features_permutation,
                                                             rank_rlsd_features_permutation,
                                                             correlation_matrix,
                                                             score_orig_features_permutation,
                                                             score_rlsd_features_permutation,
                                                             utility_collector_permutation, percentage_threshold)
            utility_collector_permutation["orig_feature_importances"] = orig_feature_importances_permutation
            utility_collector_permutation["rlsd_feature_importances"] = rlsd_feature_importances_permutation
            utility_collector_permutation["auc_orig"] = auc_orig
            utility_collector_permutation["auc_rlsd"] = auc_rlsd
            utility_collector_permutation["auc_cross"] = auc_cross
            utility_collector_permutation["f1_orig"] = f1_orig
            utility_collector_permutation["f1_rlsd"] = f1_rlsd
            utility_collector_permutation["f1_cross"] = f1_cross

            if utility_params.get("compute_shapley"):
                utility_collector_shapley = compare_features(rank_orig_features_shapley,
                                                             rank_rlsd_features_shapley,
                                                             correlation_matrix,
                                                             score_orig_features_shapley,
                                                             score_rlsd_features_shapley,
                                                             utility_collector_shapley, percentage_threshold)
                utility_collector_shapley["orig_feature_importances"] = orig_feature_importances_shapley
                utility_collector_shapley["rlsd_feature_importances"] = rlsd_feature_importances_shapley
                utility_collector_shapley["auc_orig"] = auc_orig
                utility_collector_shapley["auc_rlsd"] = auc_rlsd
                utility_collector_shapley["auc_cross"] = auc_cross
                utility_collector_shapley["f1_orig"] = f1_orig
                utility_collector_shapley["f1_rlsd"] = f1_rlsd
                utility_collector_shapley["f1_cross"] = f1_cross
            else:
                utility_collector_shapley = {}

            utility_collector_corr = {"matrix": correlation_matrix.to_numpy().tolist(),
                                      "variables": correlation_matrix.columns.tolist()}

            utility_collector = {"builtin": utility_collector_builtin,
                                 "permutation": utility_collector_permutation,
                                 "shapley": utility_collector_shapley,
                                 "correlations": utility_collector_corr}

        else:
            print("Computing feature importance for original dataset with different seeds in RF")
            final_collector = []
            for rs in random_seeds_rf:
                auc_orig_2, f1_orig_2, orig_feature_importances_builtin_2, \
                orig_feature_importances_permutation_2, orig_feature_importances_shapley_2, \
                _, X_test_orig_2, y_test_orig_2, dropped_cols_orig_2, multiclass_orig_2 = \
                    featuretools_importances(orig_df, orig_metadata, utility_params, rs)
                rank_orig_features_builtin_2 = [i[1] for i in orig_feature_importances_builtin_2]
                score_orig_features_builtin_2 = [i[0] for i in orig_feature_importances_builtin_2]

                utility_collector_builtin = compare_features(rank_orig_features_builtin, rank_orig_features_builtin_2,
                                                             score_orig_features_builtin, score_orig_features_builtin_2,
                                                             utility_collector_builtin, percentage_threshold)
                utility_collector_builtin[f"orig_feature_importances_{random_seed}"] = orig_feature_importances_builtin
                utility_collector_builtin[f"orig_feature_importances_{rs}"] = orig_feature_importances_builtin_2
                utility_collector_builtin[f"auc_orig_{random_seed}"] = auc_orig
                utility_collector_builtin[f"auc_{rs}"] = auc_orig_2
                utility_collector_builtin[f"f1_orig_{random_seed}"] = f1_orig
                utility_collector_builtin[f"f1_{rs}"] = f1_orig_2

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
                utility_collector_permutation[f"auc_orig_{random_seed}"] = auc_orig
                utility_collector_permutation[f"auc_{rs}"] = auc_orig_2
                utility_collector_permutation[f"f1_orig_{random_seed}"] = f1_orig
                utility_collector_permutation[f"f1_{rs}"] = f1_orig_2

                if utility_params.get("compute_shapley"):
                    rank_orig_features_shapley_2 = [i[1] for i in orig_feature_importances_shapley_2]
                    score_orig_features_shapley_2 = [i[0] for i in orig_feature_importances_shapley_2]
                    utility_collector_shapley = compare_features(rank_orig_features_shapley,
                                                                 rank_orig_features_shapley_2,
                                                                 score_orig_features_shapley,
                                                                 score_orig_features_shapley_2,
                                                                 utility_collector_shapley, percentage_threshold)
                    utility_collector_shapley[
                        f"orig_feature_importances_{random_seed}"] = orig_feature_importances_shapley
                    utility_collector_shapley[f"orig_feature_importances_{rs}"] = orig_feature_importances_shapley_2
                    utility_collector_shapley[f"auc_orig_{random_seed}"] = auc_orig
                    utility_collector_shapley[f"auc_{rs}"] = auc_orig_2
                    utility_collector_shapley[f"f1_orig_{random_seed}"] = f1_orig
                    utility_collector_shapley[f"f1_{rs}"] = f1_orig_2
                else:
                    utility_collector_shapley = {}

                utility_collector = {"builtin": utility_collector_builtin,
                                     "permutation": utility_collector_permutation,
                                     "shapley": utility_collector_shapley}

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
                     correlation_matrix: pd.DataFrame,
                     score_orig_features: Union[None, list] = None,
                     score_rlsd_features: Union[None, list] = None,
                     utility_collector: dict = {},
                     percentage_threshold: Union[float, None] = None):
    """Compare ranked features using different methods including
        Ranked-biased Overlap (RBO), extrapolated version of RBO,
        correlated rank similarity metrics, L2 norm and KL divergence

    Parameters
    ----------
    rank_orig_features : list
        ranked features from the original dataset
    rank_rlsd_features : list
        ranked features from the synthetic/released dataset
    correlation_matrix : pd.DataFrame
        Correlations/correlation-like measures between variables.
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

    if percentage_threshold is not None and (0 < percentage_threshold < 1) \
            and score_orig_features is not None:
        target_index = np.argmax(np.cumsum(score_orig_features) > percentage_threshold)
    else:
        target_index = len(rank_orig_features)

    # RBO - orig vs. rlsd
    orig_rlsd_sim = RankingSimilarity(rank_orig_features[:target_index],
                                      rank_rlsd_features[:target_index])
    utility_collector["rbo_0.6"] = orig_rlsd_sim.rbo(p=0.6)
    utility_collector["rbo_0.8"] = orig_rlsd_sim.rbo(p=0.8)

    # RBO - orig vs. rlsd - extrapolated version 1 (uneven + ties)
    utility_collector["rbo_ext_0.6"] = orig_rlsd_sim.rbo_ext(p=0.6)
    utility_collector["rbo_ext_0.8"] = orig_rlsd_sim.rbo_ext(p=0.8)

    # RBO - orig vs. rlsd - extrapolated version 2 (even lists)
    utility_collector["rbo_ext2_0.6"] = orig_rlsd_sim.rbo(p=0.6, ext=True)
    utility_collector["rbo_ext2_0.8"] = orig_rlsd_sim.rbo(p=0.8, ext=True)

    # Correlated rank - orig vs. rlsd - all variations
    utility_collector["corr_rank_0.6"] = orig_rlsd_sim.correlated_rank_similarity(correlation_matrix, p=0.6)
    utility_collector["corr_rank_0.8"] = orig_rlsd_sim.correlated_rank_similarity(correlation_matrix, p=0.8)
    utility_collector["corr_rank_ext2_0.6"] = orig_rlsd_sim.correlated_rank_similarity(correlation_matrix,
                                                                                       p=0.6, ext=True)
    utility_collector["corr_rank_ext2_0.8"] = orig_rlsd_sim.correlated_rank_similarity(correlation_matrix,
                                                                                       p=0.8, ext=True)

    # RBO - orig against one random permutation
    rank_rand_features = np.random.permutation(rank_orig_features[:target_index])
    orig_rand_sim = RankingSimilarity(rank_orig_features[:target_index],
                                      rank_rand_features)
    utility_collector["rbo_rand_0.6"] = orig_rand_sim.rbo(p=0.6)
    utility_collector["rbo_rand_0.8"] = orig_rand_sim.rbo(p=0.8)

    # RBO - orig against one random permutation - extrapolated version 1 (uneven + ties)
    utility_collector["rbo_rand_ext_0.6"] = orig_rand_sim.rbo_ext(p=0.6)
    utility_collector["rbo_rand_ext_0.8"] = orig_rand_sim.rbo_ext(p=0.8)

    # RBO - orig against one random permutation - extrapolated version 2 (even lists)
    utility_collector["rbo_rand_ext2_0.6"] = orig_rand_sim.rbo(p=0.6, ext=True)
    utility_collector["rbo_rand_ext2_0.8"] = orig_rand_sim.rbo(p=0.8, ext=True)

    # Correlated rank - orig against one random permutation - all variations
    utility_collector["corr_rank_rand_0.6"] = orig_rand_sim.correlated_rank_similarity(correlation_matrix, p=0.6)
    utility_collector["corr_rank_rand_0.8"] = orig_rand_sim.correlated_rank_similarity(correlation_matrix, p=0.8)
    utility_collector["corr_rank_rand_ext2_0.6"] = orig_rand_sim.correlated_rank_similarity(correlation_matrix,
                                                                                            p=0.6, ext=True)
    utility_collector["corr_rank_rand_ext2_0.8"] = orig_rand_sim.correlated_rank_similarity(correlation_matrix,
                                                                                            p=0.8, ext=True)

    # RBO - original vs. lower bound
    orig_lower_sim = RankingSimilarity(rank_orig_features[:target_index],
                                       list(reversed(rank_orig_features[:target_index])))
    utility_collector["rbo_lower_0.6"] = orig_lower_sim.rbo(p=0.6)
    utility_collector["rbo_lower_0.8"] = orig_lower_sim.rbo(p=0.8)

    # RBO - original vs. lower bound - extrapolated version 1 (uneven + ties)
    utility_collector["rbo_lower_ext_0.6"] = orig_lower_sim.rbo_ext(p=0.6)
    utility_collector["rbo_lower_ext_0.8"] = orig_lower_sim.rbo_ext(p=0.8)

    # RBO - original vs. lower bound - extrapolated version 2 (even lists)
    utility_collector["rbo_lower_ext2_0.6"] = orig_lower_sim.rbo(p=0.6, ext=True)
    utility_collector["rbo_lower_ext2_0.8"] = orig_lower_sim.rbo(p=0.8, ext=True)

    # Correlated rank - original vs. lower bound - all variations
    utility_collector["corr_rank_lower_0.6"] = orig_lower_sim.correlated_rank_similarity(correlation_matrix, p=0.6)
    utility_collector["corr_rank_lower_0.8"] = orig_lower_sim.correlated_rank_similarity(correlation_matrix, p=0.8)
    utility_collector["corr_rank_lower_ext2_0.6"] = orig_lower_sim.correlated_rank_similarity(correlation_matrix,
                                                                                              p=0.6, ext=True)
    utility_collector["corr_rank_lower_ext2_0.8"] = orig_lower_sim.correlated_rank_similarity(correlation_matrix,
                                                                                              p=0.8, ext=True)

    if score_orig_features != None and score_rlsd_features != None:
        # L2 norm
        tmp_orig_df = pd.DataFrame(score_orig_features, columns=["score_orig_features"])
        tmp_orig_df["rank_orig_features"] = rank_orig_features

        tmp_rlsd_df = pd.DataFrame(score_rlsd_features, columns=["score_rlsd_features"])
        tmp_rlsd_df["rank_rlsd_features"] = rank_rlsd_features

        tmp_rand_df = pd.DataFrame(score_orig_features, columns=["score_rand_features"])
        tmp_rand_df["rank_rand_features"] = rank_rand_features

        orig_rlsd_df = pd.merge(tmp_orig_df, tmp_rlsd_df,
                                left_on="rank_orig_features",
                                right_on="rank_rlsd_features")
        orig_rlsd_rand_df = pd.merge(orig_rlsd_df, tmp_rand_df,
                                     left_on="rank_orig_features",
                                     right_on="rank_rand_features")

        score_orig_features_array = \
            normalize(orig_rlsd_rand_df["score_orig_features"].to_numpy().reshape(-1, 1), axis=0)
        score_rlsd_features_array = \
            normalize(orig_rlsd_rand_df["score_rlsd_features"].to_numpy().reshape(-1, 1), axis=0)
        score_rand_features_array = \
            normalize(orig_rlsd_rand_df["score_rand_features"].to_numpy().reshape(-1, 1), axis=0)

        utility_collector["l2_norm"] = np.sqrt(np.sum((score_orig_features_array - score_rlsd_features_array) ** 2))
        utility_collector["l2_norm_rand"] = np.sqrt(np.sum((score_orig_features_array - score_rand_features_array) ** 2))
        utility_collector["cosine_sim"] = \
            cosine_similarity(score_orig_features_array.reshape(1, -1), score_rlsd_features_array.reshape(1, -1))[0][0]
        utility_collector["cosine_sim_rand"] = \
            cosine_similarity(score_orig_features_array.reshape(1, -1), score_rand_features_array.reshape(1, -1))[0][0]

        # KL divergence
        orig_sf = orig_rlsd_rand_df["score_orig_features"].to_numpy() + 1e-20
        rlsd_sf = orig_rlsd_rand_df["score_rlsd_features"].to_numpy() + 1e-20
        rand_sf = orig_rlsd_rand_df["score_rand_features"].to_numpy() + 1e-20
        utility_collector["kl_orig_rlsd"] = entropy(orig_sf, rlsd_sf)
        utility_collector["kl_orig_rand"] = entropy(orig_sf, rand_sf)

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
