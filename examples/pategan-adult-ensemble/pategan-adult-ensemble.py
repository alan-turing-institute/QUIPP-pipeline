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
        "dataset": "datasets/adult_dataset/adult",
        "synth-method": "PATE-GAN",
        "parameters": {
            "enabled": True,
            "num_samples_to_fit": -1,
            "num_samples_to_synthesize": 32562,
            "num_datasets_to_synthesize": 10,
            "batch_size": 128,
            "num_teachers": 50,
            "num_iters": 1000,
            "learning_rate": 0.0001,
            "columns_to_synthesize": "all",
            "epsilon": epsilon,
            "delta": 0,
            "random_state": int(random_state),
        },
        "privacy_parameters_disclosure_risk": {"enabled": False},
        "utility_parameters_classifiers": {
            "enabled": False,
            "input_columns": [
                "age",
                "workclass",
                "fnlwgt",
                "education-num",
                "marital-status",
                "occupation",
                "relationship",
                "sex",
                "capital-gain",
                "capital-loss",
                "hours",
                "native",
            ],
            "label_column": "label",
            "test_train_ratio": 0.2,
            "num_leaked_rows": 0,
            "classifier": {
                "LogisticRegression": {"mode": "main", "params_main": {"max_iter": 1000}}
            },
        },
        "utility_parameters_correlations": {"enabled": True},
        "utility_parameters_feature_importance": {
            "enabled": True,
            "label_column": "label",
            "normalized_entities": [
                {
                    "new_entity_id": "education",
                    "index": "education-num",
                    "additional_variables": ["education"],
                    "make_time_index": False,
                },
                {
                    "new_entity_id": "Workclass",
                    "index": "workclass",
                    "additional_variables": [],
                    "make_time_index": False,
                },
                {
                    "new_entity_id": "Occupation",
                    "index": "occupation",
                    "additional_variables": [],
                    "make_time_index": False,
                },
            ],
            "aggPrimitives": ["std", "min", "max", "mean", "last", "count"],
            "tranPrimitives": ["percentile"],
            "max_depth": 2,
            "features_to_exclude": ["education-num"],
            "drop_na": "columns",
            "categorical_enconding": "labels",
        },
    }


def filename_stem(i):
    return f"pategan-adult-ensemble-{i:04}"


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
    epsilons = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]

    all_params = pd.DataFrame(
        data=product(random_states, epsilons), columns=["random_state", "epsilon"]
    )

    for i, params in all_params.iterrows():
        print(dict(params))
        write_input_file(i, dict(params), force=args.force)

    if args.run:
        all_targets = [f"run-{filename_stem(i)}" for i, _ in all_params.iterrows()]
        subprocess.run(["make", "-j2", "-C../.."] + all_targets)

