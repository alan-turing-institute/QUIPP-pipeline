from typing import Callable, Any, Dict
import argparse
import json
import subprocess
import pandas as pd
from itertools import product
from pathlib import Path
from enum import Enum

QUIPP_DIR = Path(__file__).parent.parent
InputJSON = Callable[[], Dict[str, Any]]


class SynthMethod(Enum):

    PRIVBAYES = 1
    RESAMPLING = 2
    SUBSAMPLING = 3


def filename_stem(i: int, stem: str) -> str:
    return f"{stem}-{i:04}"


def input_path(i: int, stem: str):
    if not (QUIPP_DIR / "run-inputs").exists():
        print("Could not find run-inputs dir")
        exit()
    return QUIPP_DIR / "run-inputs" / f"{filename_stem(i, stem)}.json"


def write_input_file(
    i: int,
    filestem: str,
    input_json: InputJSON,
    params: Dict[str, Any],
    force: bool = False,
):
    fname = input_path(i, filestem)
    run_input = json.dumps(input_json(**params), indent=4)
    if force or not fname.exists():
        print(f"Writing {fname}")
        print(fname.exists())
        with open(fname, "w") as input_file:
            input_file.write(run_input)


def read_json(fname):
    with open(fname) as f:
        return json.load(f)


def handle_cmdline_args(synth_method: SynthMethod):
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

    if synth_method == SynthMethod.PRIVBAYES:

        parser.add_argument(
            "-e",
            "--epsilons",
            dest="epsilons",
            required=True,
            help="Define list of epsilons for the requested run",
        )

        parser.add_argument(
            "-k",
            "--parents",
            dest="k",
            required=True,
            type=int,
            help="Define k (number of parents) for the requested run",
        )

    if synth_method == SynthMethod.SUBSAMPLING:

        parser.add_argument(
            "-s",
            "--sample-fractions",
            dest="sample_fracs",
            required=True,
            help="The list of fraction of samples used",
        )

    args = parser.parse_args()
    return args


def run(
    input_json: Callable[[], Dict[str, Any]], filestem: str, synth_method: SynthMethod
):

    args = handle_cmdline_args(synth_method)

    random_states = range(args.nreplicas)

    if synth_method == SynthMethod.PRIVBAYES:
        all_params = pd.DataFrame(
            data=product(
                random_states,
                map(float, args.epsilons.strip("[]").split(",")),
                [args.k],
            ),
            columns=["random_state", "epsilon", "k"],
        )
    elif synth_method == SynthMethod.RESAMPLING:
        all_params = pd.DataFrame(data=random_states, columns=["random_state"])
    else:
        raise TypeError("Not implemented for this type")

    for i, params in all_params.iterrows():
        print(dict(params))
        write_input_file(i, filestem, input_json, dict(params), force=args.force)

    if args.run:
        all_targets = [
            f"run-{filename_stem(i, filestem)}" for i, _ in all_params.iterrows()
        ]
        subprocess.run(["make", "-k", "-j72",] + all_targets)
