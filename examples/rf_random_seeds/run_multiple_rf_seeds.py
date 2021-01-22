# Script to calculate feature importance metric using different Random Forest training seeds.
# It only requires one run-input file as it does not do synthesis.
# Should be run from the root QUIPP directory with arguments:
# -i [run-input-file] -o [synth-output-dir] -s [list-of-seeds-for-RF]

import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "metrics/utilities"))
from utils import handle_cmdline_args_rf_seeds, extract_parameters, find_column_types

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "metrics/utility-metrics"))
from rbo import RankingSimilarity
from feature_importance import feature_importance_metrics


def main():
    # process command line arguments
    args = handle_cmdline_args_rf_seeds()

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
    output_file_json = os.path.join(path_released_ds, "utility_feature_importance_rf_seeds.json")

    # calculate and save feature importance metrics
    feature_importance_metrics(
        path_original_ds,
        path_original_meta,
        None,
        utility_params_ft,
        output_file_json,
        None,
        random_seed,
        map(int, args.seeds.strip('[]').split(','))
    )


if __name__ == "__main__":
    main()
