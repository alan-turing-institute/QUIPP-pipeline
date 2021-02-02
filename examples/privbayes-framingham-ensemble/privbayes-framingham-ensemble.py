import argparse
import json
import matplotlib.pyplot as plt
import subprocess
import seaborn as sns
import pandas as pd
from itertools import product
from pathlib import Path


def input_json(random_state, epsilon, k):
    return {
        "enabled": True,
        "label_column": "TenYearCHD",
        "aggPrimitives": [
            "mean", "max", "min"
        ],
        "tranPrimitives": [],
        "max_depth": 2,
        "features_to_exclude": [],
        "drop_na": "rows",
        "normalized_entities": [
            {"new_entity_id": "edu",
             "index": "education",
             "make_time_index": False
            },
            {"new_entity_id": "smoking",
             "index": "cigsPerDay",
             "additional_variables": ["currentSmoker"],
             "make_time_index": False
            },
            {"new_entity_id": "sex",
             "index": "male",
             "make_time_index": False
            },
            {"new_entity_id": "lifeyears",
             "index": "age",
             "make_time_index": False
            },
            {"new_entity_id": "hypertension",
             "index": "prevalentHyp",
             "make_time_index": False
            },
            {"new_entity_id": "diabetic",
             "index": "diabetes",
             "make_time_index": False
            }
            ],
        "categorical_enconding": "labels",
        "compute_shapley": True,
        "skip_feature_engineering": False
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

    parser.add_argument(
        "-e",
        "--epsilon",
        dest="epsilon",
        required=True,
        type=float,
        help="Define epsilon for the requested run",
    )

    parser.add_argument(
        "-k",
        "--parents",
        dest="k",
        required=True,
        type=int,
        help="Define k (number of parents) for the requested run",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = handle_cmdline_args()

    random_states = range(args.nreplicas)

    all_params = pd.DataFrame(
        data=product(random_states, [args.epsilon], [args.k]), columns=["random_state", "epsilon", "k"]
    )

    for i, params in all_params.iterrows():
        print(dict(params))
        write_input_file(i, dict(params), force=args.force)

    if args.run:
        all_targets = [f"run-{filename_stem(i)}" for i, _ in all_params.iterrows()]
        subprocess.run(["make", "-k", "-j72", "-C../.."] + all_targets)
