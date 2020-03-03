## prepare_data.py 

import pandas as pd
import json
import argparse
import sys
import os

def handle_cmdline_args():
    parser = argparse.ArgumentParser(
        description='Read input files and output file for run of sgf.')

    parser.add_argument('parameter_json', help='The json containing the synthesis parameters')

    parser.add_argument('data_path_prefix', help='The prefix of the path to the input data, relative to QUIPP-pipeline root (append .csv and .json to get the data files)')

    args = parser.parse_args()

    return args


def write_cfg(parameters_dict):
    # short name for the string interpolation
    d = parameters_dict
    print(d)
    cfg = f"""[all]
mechanism=seedbased
attrs={d['attrs']}
gamma={d['gamma']}
omega={d['omega']}
ncomp={d['ncomp']}
ndist={d['ndist']}
max_ps=10000
max_check_ps=0
workdir={d['workdir']}
dataprefix={d['dataprefix']}
count={d['count']}
runtime=3600
"""
    with open("my.cfg", "w") as f:
        f.write(cfg)


def main():
    args = handle_cmdline_args()
    
    parameter_json_path = args.parameter_json
    with open(parameter_json_path) as f:
        parameter_json = json.load(f)
        
    method_params = parameter_json['parameters']

    data = pd.read_csv(args.data_path_prefix + ".csv")
    
    ## split

    if (method_params['num_samples_to_fit'] == -1):
        count = data.shape[0]
    else:
        count = method_params['num_samples_to_fit']

    parameters_dict = {'attrs': len(data.columns),
                      'gamma': method_params['gamma'],
                      'omega': method_params['omega'],
                      'ncomp': method_params['ncomp'],
                      'ndist': method_params['ndist'],
                      'workdir': os.getcwd(),
                      'dataprefix': os.path.basename(args.data_path_prefix),
                      'count': count}

    write_cfg(parameters_dict)


if __name__=="__main__":
    main()
