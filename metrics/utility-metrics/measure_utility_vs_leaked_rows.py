#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Run sklearn_classifiers.py for different number of leaked rows and compare the utility metrics.
XXX This code is still under-development.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os

# ------ input
# run sklearn_classifiers.py? 
# if the results are already created, you can set this to False
# and set plot_results = True
run_utility_measurements = True
plot_results = True
# parent directory to store results
output_dir = "results"
# files will be saved in this path: {output_dir}/{output_file_prefix}_{number of leaked rows}.json
output_file_prefix = "utility_leaked"
# number of leaked rows in each iteration
leaked_rows_list = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
# ------

if run_utility_measurements:
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    for leaked_rows in leaked_rows_list:
        base_python_code = f''' 
        python sklearn_classifiers.py \
            --path_original_ds="../generator-outputs/odi-nhs-ae/hospital_ae_data_deidentify.csv" \
            --path_original_meta="../generator-outputs/odi-nhs-ae/hospital_ae_data_deidentify.json" \
            --path_released_ds="../synth-output/synthpop-example-3-samples/synthetic_data_1.csv" \
            --input_columns='["Time in A&E (mins)"]' \
            --label_column="Age bracket" \
            --test_train_ratio="0.2" \
            --output_file_json="{output_dir}/{output_file_prefix}_{leaked_rows}.json" \
            --num_leaked_rows={leaked_rows}
        '''
        os.system(base_python_code)

if plot_results:
    all_results = {}
    for leaked_rows in leaked_rows_list:
        utility_collector = json.load(open(f"{output_dir}/{output_file_prefix}_{leaked_rows}.json", "r"))
        for one_measure in utility_collector:
            # difference between F1-scores of r_o against o_o, i.e.
            # model trained on released dataset and tested on original dataset (r_o)
            # model trained on original dataset and tested on original dataset (o_o)
            f1_score_diff = utility_collector[one_measure]["f1_r_o"] - utility_collector[one_measure]["f1_o_o"]
            if not one_measure in all_results:
                all_results[one_measure] = [[leaked_rows, f1_score_diff]]
            else:
                all_results[one_measure].append([leaked_rows, f1_score_diff])
    
    for one_method in all_results:
        results_arr = np.array(all_results[one_method])
        plt.plot(results_arr[:, 0], results_arr[:, 1], label=one_method, marker='o', lw=2)
    plt.xlabel("#leaked rows", size=22)
    plt.ylabel("F1-score (synthetic - original)", size=22)
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.grid()
    plt.legend(fontsize=18)
    plt.show()
