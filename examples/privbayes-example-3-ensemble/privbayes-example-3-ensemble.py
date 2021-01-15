import argparse
import json
import subprocess
from pathlib import Path


def input_json(seed):
    return {
        "enabled": True,
        "dataset": "datasets/appointment_noshows/KaggleV2-May-2016-cleaned",
        "synth-method": "PrivBayes",
        "parameters": {
            "enabled": True,
            "num_samples_to_synthesize": 110527,
            "random_state": seed,
            "category_threshold": 20,
            "epsilon": 0.2,
            "k": 1,
            "keys": ["appointment_id"],
            "histogram_bins": 10,
            "preconfigured_bn": {},
        },
        "privacy_parameters_disclosure_risk": {
            "enabled": True,
            "num_samples_intruder": 2000,
            "vars_intruder": ["gender", "age", "neighborhood"],
        },
        "utility_parameters_classifiers": {
            "enabled": True,
            "input_columns": ["gender", "neighborhood", "scholarship", "handicap"],
            "label_column": "no_show",
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
            "entity_index": "appointment_id",
            "time_index": "scheduled_time",
            "label_column": "no_show",
            "secondary_time_index": {"appointment_day": ["no_show", "sms_received"]},
            "normalized_entities": [
                {
                    "new_entity_id": "patients",
                    "index": "patient_id",
                    "additional_variables": [
                        "scholarship",
                        "hypertension",
                        "diabetes",
                        "alcoholism",
                        "handicap",
                    ],
                },
                {
                    "new_entity_id": "locations",
                    "index": "neighborhood",
                    "make_time_index": False,
                },
                {"new_entity_id": "ages", "index": "age", "make_time_index": False},
                {
                    "new_entity_id": "genders",
                    "index": "gender",
                    "make_time_index": False,
                },
            ],
            "features_to_exclude": ["patient_id"],
        },
    }


def input_file_paths(n):
    return [
        Path(f"../../run-inputs/privbayes-example-3-ensemble-{i:04}.json")
        for i in range(n)
    ]


def write_input_files(n):
    for i, fname in enumerate(input_file_paths(n)):
        run_input = json.dumps(input_json(i))
        with open(fname, "w") as input_file:
            input_file.write(run_input)


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

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = handle_cmdline_args()
    write_input_files(args.nreplicas)

    if args.run:
        subprocess.run(["make", "-C../.."])
