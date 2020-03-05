## prepare_data.py 

import numpy as np
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


def write_cfg(parameters_dict, filename):
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
max_ps=0
max_check_ps=0
workdir={d['workdir']}
dataprefix={d['dataprefix']}
count={d['count']}
runtime=3600
"""
    with open(filename, "w") as f:
        f.write(cfg)


def write_dag(corr, filename):
    # short name for the string interpolation
    dag = f"""8, {corr}
0
0
0
0
0
0
0
"""
    with open(filename, "w") as f:
        f.write(dag)


def write_order(filename):
    # short name for the string interpolation
    order = f"""2
3
4
5
6
7
8
1
"""
    with open(filename, "w") as f:
        f.write(order)





def write_attrs(data, filename):
    with open(filename, "w") as f:
        for col in data.columns:
            line = ",".join([col, *map(str, np.unique(data[col]))]) + "\n"
            f.write(line)


def write_grps(data, filename):
    with open(filename, "w") as f:
        for col in data.columns:
            line = ",".join([*map(str, np.unique(data[col]))]) + "\n"
            f.write(line)


def main():
    args = handle_cmdline_args()
    
    parameter_json_path = args.parameter_json
    with open(parameter_json_path) as f:
        parameter_json = json.load(f)
        
    method_params = parameter_json['parameters']

    data = pd.read_csv(args.data_path_prefix + ".csv")
    
    ## use the basename of the data file path prefix, relative to the
    ## working directory
    dataprefix = os.path.join(os.getcwd(),
                              os.path.basename(args.data_path_prefix))


    ## split the data into "stats" (training) and "records" (seeding)
    stats = data.sample(frac = 0.5,
                        random_state = method_params['random_state'])
    records = data.drop(stats.index)

    stats.to_csv(dataprefix + "_stats.csv", index=False)
    records.to_csv(dataprefix + "_records.csv", index=False)


    ## write config file
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
                      'dataprefix': dataprefix,
                      'count': count}

    write_cfg(parameters_dict, "my.cfg")


    ## write "attrs" and "grps"
    write_attrs(data, dataprefix + "_attrs.csv")
    write_grps(data, dataprefix + "_grps.csv")
    write_dag(0.95, dataprefix + "_dag.csv")
    write_order(dataprefix + "_order.csv")


if __name__=="__main__":
    main()
