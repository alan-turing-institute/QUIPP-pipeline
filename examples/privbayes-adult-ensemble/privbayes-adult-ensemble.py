import argparse
import json
import matplotlib.pyplot as plt
import subprocess
import pandas as pd
from itertools import product
from pathlib import Path


def input_json(random_state, epsilon):
    return {
        "enabled": True,
        "dataset": "datasets/adult_dataset/adult",
        "synth-method": "PrivBayes",
        "parameters": {
            "enabled": True,
            "num_samples_to_synthesize": 32562,
            "random_state": random_state,
            "category_threshold": 20,
            "epsilon": epsilon,
            "k": 3,
            "keys": ["appointment_id"],
            "histogram_bins": 10,
            "preconfigured_bn": {},
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
                "LogisticRegression": {
                    "mode": "main",
                    "params_main": {"max_iter": 1000},
                }
            },
        },
        "utility_parameters_correlations": {"enabled": True},
        "utility_parameters_feature_importance": {
            "enabled": True,
            "label_column": "label",
            "normalized_entities": [],
        },
    }

def filename_stem(i):
    return f"privbayes-adult-ensemble-{i:04}"


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
        return json.loads(f.read())


def collect_results(all_params):
    results = pd.DataFrame(
        [read_json(feature_importance_path(i)) for i, _ in all_params.iterrows()]
    )

    return all_params.join(results)


def plot_results(df):
    df.boxplot(by="epsilon", column="rbo_0.8")
    plt.ylabel("RBO(feature importances)")
    plt.tight_layout()
    plt.savefig("rbo.pdf")


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
    epsilons = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    all_params = pd.DataFrame(
        data=product(random_states, epsilons), columns=["random_state", "epsilon"]
    )

    for i, params in all_params.iterrows():
        print(dict(params))
        write_input_file(i, dict(params), force=args.force)

    if args.run:
        all_targets = [f"run-{filename_stem(i)}" for i, _ in all_params.iterrows()]
        subprocess.run(["make", "-j", "-C../.."] + all_targets)

        df = collect_results(all_params)
        plot_results(df)
