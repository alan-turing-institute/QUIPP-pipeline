#!/usr/bin/env python

"""
Generate a report from sklearn_classifier outputs
"""

import codecs
from datetime import datetime
import matplotlib.pyplot as plt
import os
import pandas as pd
from plotting_tools import plot_util_confusion_matrix
from sklearn_classifiers import printMetric
import json

list_items2print = [
    {
        "filename": "utility_o_o.json",
        "action": "printMetric",
        "title2print": "Trained on original and tested on original"
    },
    {
        "filename": "utility_confusion_o_o.json", 
        "prefix": "o_o",
        "action": "plot_confusion"
    },
    {
        "filename": "utility_r_o.json",
        "action": "printMetric",
        "title2print": "Trained on released and tested on original"
    },
    {
        "filename": "utility_confusion_r_o.json", 
        "prefix": "r_o",
        "action": "plot_confusion"
    }
]

def print_header():
    myheader = """
<html>

<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
* {
  box-sizing: border-box;
}

/* Create two equal columns that floats next to each other */
.column {
  float: left;
  width: 50%;
  padding: 10px;
}

/* Clear floats after the columns */
.row:after {
  content: "";
  display: table;
  clear: both;
}
</style>
</head>

<body>"""
    return myheader

def report(path2synth_output):
    # --- generate report
    curr_time = datetime.now()
    curr_time_str = "{}-{}-{}_{}-{}-{}".format(curr_time.year, 
                                               curr_time.month,
                                               curr_time.day,
                                               curr_time.hour,
                                               curr_time.minute,
                                               curr_time.second)
    
    curr_time_str_print = "{}-{}-{} {}:{}:{}".format(curr_time.year, 
                                                     curr_time.month,
                                                     curr_time.day,
                                                     curr_time.hour,
                                                     curr_time.minute,
                                                     curr_time.second)
    
    par_dir = f"./report_{curr_time_str}"
    par_dir = os.path.join(path2synth_output, par_dir)
    
    if not os.path.isdir(par_dir):
        os.makedirs(par_dir)
    
    filename = os.path.join(par_dir, "report.html")
    f = open(filename, 'w')
    
    message = print_header()
    message += "<h1>Report" + "(" + curr_time_str_print + ")" + "</h1>"
    
    # --- Plot overall differences
    fio = open(os.path.join(path2synth_output, "utility_overall_diff.json")).read()
    dict_read = json.loads(fio)
    message += "<h2>Overall difference</h2><p>"
    # extract metric names/values
    metric_name = []
    metric_value = []
    for k_metric, v_metric in dict_read.items():
        for k_value, v_value in v_metric.items():
            metric_name.append(f"{k_metric}_{k_value}")
            metric_value.append(v_value*100)
            #message += f"{k_metric} ({k_value}): {v_value*100:.3f}" + "<br/>"
    #message += "<br/>"
    path2image = os.path.abspath(os.path.join(par_dir, "figs", "overall_diffs.png"))
    
    if not os.path.isdir(os.path.dirname(path2image)):
        os.makedirs(os.path.dirname(path2image))
    
    plt.figure(figsize=(10, 5))
    plt.bar(metric_name, metric_value, color='k')
    plt.ylabel("Difference (%)", size=18)
    plt.title("Overall difference", size=24)
    plt.xticks(size=18, rotation=90)
    plt.yticks(size=18)
    plt.grid()
    plt.savefig(path2image, bbox_inches="tight")
    message += f'<img src={path2image} alt="" width="800" align="middle"><br />' 
    message += "<hr>"
    
    # --- Plot differences in each method
    fio = open(os.path.join(path2synth_output, "utility_diff.json")).read()
    dict_read = json.loads(fio)
    message += "<h2>Differences in each method</h2><p>"
    # convert the nested dictionary to a pandas dataframe
    df_rd = pd.concat({k: pd.DataFrame(v).T for k, v in dict_read.items()}, axis=0)
    list_methods = df_rd.index.get_level_values(0).to_list()
    # list of unique methods
    list_uniq_methods = []
    for one_method in list_methods:
        if one_method not in list_uniq_methods:
            list_uniq_methods.append(one_method)
    
    # extract precision/recall/F1 scores
    prec_values = []
    recall_values = []
    f1_values = []
    for one_method in list_uniq_methods:
        prec_values.append(df_rd.loc[one_method, "precision"]["weighted"]*100.)
        recall_values.append(df_rd.loc[one_method, "recall"]["weighted"]*100.)
        f1_values.append(df_rd.loc[one_method, "f1"]["weighted"]*100.)
    
    plt.figure(figsize=(10, 5))
    plt.plot(prec_values, c="b", lw=3, marker="o", label="Precision (weighted)")
    plt.plot(recall_values, c="r", lw=3, marker="o", label="Recall (weighted)")
    plt.plot(f1_values, c="k", lw=3, marker="o", label="F1 (weighted)")
    plt.ylabel("Difference (%)", size=32)
    
    path2image = os.path.abspath(os.path.join(par_dir, "figs", "diffs.png"))
    
    if not os.path.isdir(os.path.dirname(path2image)):
        os.makedirs(os.path.dirname(path2image))
    
    plt.xticks(range(len(list_uniq_methods)), list_uniq_methods, size=24, rotation=90)
    plt.yticks(size=24)
    plt.legend(fontsize=24, bbox_to_anchor=(1.04,1))
    plt.grid()
    plt.savefig(path2image, bbox_inches="tight")
    message += f'<img src={path2image} alt="" width="1000" align="middle"><br />' 
    message += "<hr>"
    
    # Add details of each method / plot confusion matrix
    # The details will be printed/plotted in two columns
    message += '<div class="column"><p>'
    # Are we writing on the right column?
    on_right_window = False
    for one_item in list_items2print:
        # XXX fragile, we check if r_o is in the filename
        # If yes, we go to the right column
        if "r_o" in one_item["filename"] and not on_right_window:
            message += "</p></div>"
            message += '<div class="column"><p>'
            on_right_window = True
    
        if one_item["action"] == "printMetric":
            fio = open(os.path.join(path2synth_output, one_item["filename"])).read()
            dict_read = json.loads(fio)
            msg = printMetric(dict_read, title=one_item["title2print"])
            message += msg
    
        if one_item["action"] == "plot_confusion":
            plt_names = plot_util_confusion_matrix(os.path.join(path2synth_output, one_item["filename"]), 
                                       method_names=None, prefix=one_item["prefix"], 
                                       save_dir=os.path.join(par_dir, "figs")
                                       )
            for plt_name in plt_names:
                message += "<hr>"
                message += f'<img src={plt_name} alt="" width="500"><br />' 
    
    message += "</body></html>"
    f.write(message)
    f.close()

if __name__ == "__main__":
    # XXX move this to the input file
    path2synth_output = "../synth-output/polish-synthpop-dr-4" 
    report(path2synth_output)