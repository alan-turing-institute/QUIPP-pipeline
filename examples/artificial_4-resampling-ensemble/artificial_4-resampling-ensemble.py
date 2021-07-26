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
        "dataset": "generator-outputs/artificial/artificial_4",
        "synth-method": "synthpop",
        "parameters": {
            "enabled": True,
            "num_samples_to_fit": -1,
            "num_samples_to_synthesize": -1,
            "num_datasets_to_synthesize": 1,
            "random_state": int(random_state),
            "vars_sequence": [1, 2, 3, 4, 5, 6],
            "synthesis_methods": [
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
            "label_column": "Label",
            "normalized_entities": [],
            "max_depth": 2,
            "aggPrimitives": [],
            "tranPrimitives": [],
            "drop_na": True,
            "drop_full_na_columns": False,
            "compute_shapley": True,
            "skip_feature_engineering": True,
            "categorical_enconding": "labels"
        },
    }

    

def filename_stem(i):
    return f"artificial_4-resampling-ensemble-{i:04}"


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
