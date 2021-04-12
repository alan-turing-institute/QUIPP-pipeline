import argparse
import json
import matplotlib.pyplot as plt
import subprocess
import pandas as pd
from itertools import product
from pathlib import Path

def input_json(random_state):
    return {
        "enabled": True,
        "dataset": "generator-outputs/household_poverty/train_cleaned_large",
        "synth-method": "synthpop",
        "parameters": {
            "enabled": True,
            "num_samples_to_fit": -1,
            "num_samples_to_synthesize": -1,
            "num_datasets_to_synthesize": 1,
            "random_state": int(random_state),
            "vars_sequence": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                              17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                              30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
            "synthesis_methods": [
                "",
                "sample",
                "sample",
                "sample",
                "sample",
                "sample",
                "sample",
                "sample",
                "sample",
                "sample",
                "sample",
                "sample",
                "sample",
                "sample",
                "sample",
                "sample",
                "sample",
                "sample",
                "sample",
                "sample",
                "sample",
                "sample",
                "sample",
                "sample",
                "sample",
                "sample",
                "sample",
                "sample",
                "sample",
                "sample",
                "sample",
                "sample",
                "sample",
                "sample",
                "sample",
                "sample",
                "sample",
                "sample",
                "sample"
            ],
            "proper": False,
            "tree_minbucket": 1,
        },
        "privacy_parameters_disclosure_risk": {
            "enabled": False,
            "num_samples_intruder": 5000,
            "vars_intruder": ["gender", "age", "neighborhood"],
        },
        "utility_parameters_classifiers": {
            "enabled": False,
            "classifier": {
                "LogisticRegression": {"mode": "main", "params_main": {"max_iter": 1000}}
            },
        },
        "utility_parameters_correlations": {"enabled": False},
        "utility_parameters_feature_importance": {
            "enabled": True,
            "entity_index": "Id",
            "label_column": "Target",
            "normalized_entities": [
                {"new_entity_id": "household",
                 "index": "idhogar",
                 "additional_variables": ["pared", "piso", "energcocinar", "cielorazo",
                                          "epared", "etecho", "eviv",
                                          "rooms", "r4m1", "r4m2",
                                          "r4m3", "r4t1", "r4t2", "r4t3",
                                          "hogar_nin", "bedrooms", "qmobilephone",
                                          "dependency", "edjefe", "meaneduc",
                                          "overcrowding", "hhsize",
                                          "television", "SQBdependency", "Target"]
                 }
            ],
            "max_depth": 2,
            "aggPrimitives": ["min", "max", "count", "mode", "num_unique", "std", "sum"],
            "target_entity": "household",
            "drop_na": True,
            "drop_full_na_columns": True,
            "na_thresh": 0.30,
            "compute_shapley": True,
            "skip_feature_engineering": False,
            "features_to_exclude": ["idhogar"],
            "filter_hh": True
        }
    }


def filename_stem(i):
    return f"household_large-resampling-ensemble-{i:04}"


def input_path(i):
    return Path(f"../../run-inputs/{filename_stem(i)}.json")


def feature_importance_path(i):
    return Path(
        f"../../synth-output/{filename_stem(i)}/utility_feature_importance.json"
    )


def write_input_file(i, params, force=False):
    fname = input_path(i)
    run_input = json.dumps(input_json(**params), indent=4)
    if force or not fname.exists():
        print(f"Writing {fname}")
        with open(fname, "w") as input_file:
            input_file.write(run_input)


def read_json(fname):
    with open(fname) as f:
        return json.load(f)


def handle_cmdline_args():
    parser = argparse.ArgumentParser(
        description="Generate (optionally run and postprocess) an ensemble of run inputs"
    )

    parser.add_argument(
        "-n",
        "--num-replicas",
        dest="nreplicas",
        required=True,
        type=int,
        help="The number of replicas to generate",
    )

    parser.add_argument(
        "-r",
        "--run",
        default=False,
        action="store_true",
        help="Run (via make) and postprocess?",
    )

    parser.add_argument(
        "-f",
        "--force-write",
        dest="force",
        default=False,
        action="store_true",
        help="Write out input files, even if they exist",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = handle_cmdline_args()

    random_states = range(args.nreplicas)

    all_params = pd.DataFrame(
        data=random_states, columns=["random_state"]
    )

    for i, params in all_params.iterrows():
        print(dict(params))
        write_input_file(i, dict(params), force=args.force)

    if args.run:
        all_targets = [f"run-{filename_stem(i)}" for i, _ in all_params.iterrows()]
        subprocess.run(["make", "-j", "-C../.."] + all_targets)
