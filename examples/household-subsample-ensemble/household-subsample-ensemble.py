import argparse
import json
import matplotlib.pyplot as plt
import subprocess
import pandas as pd
from itertools import product
from pathlib import Path

def input_json(random_state, sample_frac):
    return {
        "enabled": True,
        "dataset": "generator-outputs/household_poverty/train_cleaned",
        "synth-method": "subsample",
        "parameters": {
            "enabled": True,
            "frac_samples_to_synthesize": sample_frac,
            "random_state": int(random_state),
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
                 "additional_variables": ["hacdor", "hacapo", "v14a", "refrig", "paredblolad", "paredzocalo",
                                          "paredpreb", "pisocemento", "pareddes", "paredmad",
                                          "paredzinc", "paredfibras", "paredother", "pisomoscer", "pisoother",
                                          "pisonatur", "pisonotiene", "pisomadera",
                                          "techozinc", "techoentrepiso", "techocane", "techootro", "cielorazo",
                                          "abastaguadentro", "abastaguafuera", "abastaguano",
                                          "public", "planpri", "noelec", "coopele", "sanitario1",
                                          "sanitario2", "sanitario3", "sanitario5", "sanitario6",
                                          "energcocinar1", "energcocinar2", "energcocinar3", "energcocinar4",
                                          "elimbasu1", "elimbasu2", "elimbasu3", "elimbasu4",
                                          "elimbasu5", "elimbasu6", "epared1", "epared2", "epared3",
                                          "etecho1", "etecho2", "etecho3", "eviv1", "eviv2", "eviv3",
                                          "tipovivi1", "tipovivi2", "tipovivi3", "tipovivi4", "tipovivi5",
                                          "computer", "television", "lugar1", "lugar2", "lugar3",
                                          "lugar4", "lugar5", "lugar6", "area1", "area2",
                                          "rooms", "r4h1", "r4h2", "r4h3", "r4m1", "r4m2", "r4m3", "r4t1", "r4t2",
                                          "r4t3", "v18q1", "tamhog", "tamviv", "hhsize", "hogar_nin",
                                          "hogar_adul", "hogar_mayor", "hogar_total", "bedrooms", "qmobilephone",
                                          "v2a1", "dependency", "edjefe", "edjefa", "meaneduc", "overcrowding",
                                          "Target"]
                 }
            ],
            "max_depth": 2,
            "aggPrimitives": ["min", "max", "count", "mode", "num_unique", "std", "sum"],
            "target_entity": "household",
            "drop_na": True,
            "drop_na_columns_thresh": True,
            "na_thresh": 0.30,
            "compute_shapley": True,
            "skip_feature_engineering": False,
            "features_to_exclude": ["idhogar"],
            "filter_hh": True
        }
    }


def filename_stem(i):
    return f"household-subsample-ensemble-{i:04}"


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
        "-s",
        "--sample-fractions",
        dest="sample_fracs",
        required=True,
        help="The list of fractions of samples used",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = handle_cmdline_args()

    random_states = range(args.nreplicas)

    all_params = pd.DataFrame(
        data=product(random_states, map(float, args.sample_fracs.strip('[]').split(','))), columns=["random_state", "sample_frac"]
    )

    for i, params in all_params.iterrows():
        print(dict(params))
        write_input_file(i, dict(params), force=args.force)

    if args.run:
        all_targets = [f"run-{filename_stem(i)}" for i, _ in all_params.iterrows()]
        subprocess.run(["make", "-j72", "-C../.."] + all_targets)