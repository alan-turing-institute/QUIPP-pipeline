# This script reads the "synthesis parameters" json, and dispatches to
# the synth-method therein
#
# Should be run from within the QUIPP-pipeline root directory

import json
import argparse
import sys
import os

def handle_cmdline_args():
    """Return an object with attributes 'infile' and 'outfile', after
handling the command line arguments"""

    parser = argparse.ArgumentParser(
        description='Generate synthetic data from a specification in a json '
        'file using the "synth-method" described in the json file.  ')

    parser.add_argument(
        '-i', dest='infile', required=True,
        help='The input json file. Must contain a "synth-method" property')

    parser.add_argument(
        '-o', dest='outfile_prefix', required=True, help='The prefix of the output paths (data json and csv), relative to the QUIPP-pipeline root directory')

    args = parser.parse_args()
    return args


def main():
    # read command line options
    args = handle_cmdline_args()

    with open(args.infile) as f:
        synth_params = json.load(f)

    if not (synth_params["enabled"] and synth_params['parameters']['enabled']):
        return

    synth_method = synth_params["synth-method"]
    dataset = synth_params["dataset"]

    synth_method_cmd = os.path.abspath(os.path.join(
        "synth-methods", synth_method, "run"))

    dataset_path_base = os.path.abspath(dataset)

    input_json = os.path.abspath(args.infile)

    os.chdir(os.path.dirname(args.outfile_prefix))

    os.execv(synth_method_cmd,
             ["run", input_json,
              dataset_path_base,
              os.path.basename(args.outfile_prefix)])


if __name__=='__main__':
    main()

