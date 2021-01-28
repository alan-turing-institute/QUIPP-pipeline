import argparse
import json
import matplotlib.pyplot as plt
import subprocess
import seaborn as sns
import pandas as pd
from itertools import product
from pathlib import Path


def input_json(random_state, epsilon):
    return {
        "enabled": True,
        "dataset": "datasets/framingham/framingham_cleaned",
        "synth-method": "PrivBayes",
        "parameters": {
            "enabled": True,
            "num_samples_to_synthesize": 4240,
            "random_state": int(random_state),
            "category_threshold": 20,
            "epsilon": epsilon,
            "k": 3,
            "keys": [],
            "histogram_bins": 10,
            "preconfigured_bn": {},
            "save_description": False
        },
        "privacy_parameters_disclosure_risk": {"enabled": False},
        "utility_parameters_classifiers": {
            "enabled": False,
            "classifier": {
                "LogisticRegression": {"mode": "main", "params_main": {"max_iter": 1000}}
            },
        },
        "utility_parameters_correlations": {"enabled": True},
        "utility_parameters_feature_importance": {
            "enabled": True,
            "label_column": "TenYearCHD",
            "max_depth": 2,
            "features_to_exclude": ["currentSmoker"],
            "drop_na": "columns",
            "categorical_enconding": "labels",
            "normalize_entities": [],
            "compute_shapley": False
        },
    }


def filename_stem(i):
    return f"privbayes-framingham-ensemble-{i:04}"


def input_path(i):
    return Path(f"../../run-inputs/{filename_stem(i)}.json")


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
    epsilons = [0.001, 0.1, 1.0, 10.0]

    all_params = pd.DataFrame(
        data=product(random_states, epsilons), columns=["random_state", "epsilon"]
    )

    for i, params in all_params.iterrows():
        print(dict(params))
        write_input_file(i, dict(params), force=args.force)

    if args.run:
        all_targets = [f"run-{filename_stem(i)}" for i, _ in all_params.iterrows()]
        subprocess.run(["make", "-k", "-j72", "-C../.."] + all_targets)
