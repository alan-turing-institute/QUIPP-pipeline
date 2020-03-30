import json
import os
import subprocess
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse

out_path = "../run-inputs/"
synth_path = "../synth-output/"


def handle_cmdline_args():
    """Return an object with attribute 'indirectory', after
    handling the command line arguments"""

    parser = argparse.ArgumentParser(
        description='Run pipeline for each .json files in the input directory '
                    'and plot the results. ')

    parser.add_argument(
        '-i', dest='indirectory', required=False,
        default="parameter_files/generated-synthpop-dr",
        help='The directory containing all input json files.')

    args = parser.parse_args()
    return args


def copy_parameter_files(indirectory):
    """Cope all .json parameter files from indirectory to out_path (global
    variable)."""
    f_list = os.listdir(path=indirectory)
    for f in f_list:
        shutil.copyfile(os.path.join(indirectory, f), os.path.join(out_path, f))


def run_pipeline():
    """Runs QUiPP pipeline for all enabled .json files in run-inputs
    Note: Can take several minutes depending on number of files and
    method used. Not all output is printed to the console. If you want to see
    all the output, you can run make from bash."""
    cmd = '''
    cd ..
    make   
    '''
    subprocess.check_output(cmd, shell=True)


def main(*args):

    args = handle_cmdline_args()
    indirectory = args.indirectory

    #copy_parameter_files(indirectory)
    #run_pipeline()

    f_list = os.listdir(path=indirectory)
    privacy_scores = []
    utility_scores = []

    for f in [os.path.splitext(name)[0] for name in f_list]:

        with open(os.path.join(synth_path, f, "privacy_metric_disclosure_risk.json")) as privacy_file:
            privacy_dict = json.load(privacy_file)
        privacy_scores.append(privacy_dict["EMRi_norm"])

        with open(os.path.join(synth_path, f, "utility_metric_sklearn.json")) as utility_file:
            utility_dict = json.load(utility_file)
            print(utility_file)
        utility_scores.append(utility_dict["Overall"]["f1_diff"])

    print(f"Privacy scores: {1-np.array(privacy_scores)}")
    print(f"Utility scores: {1-np.array(utility_scores)}")

    sns.set_palette("hls", 8)
    ax = sns.scatterplot(x=1-np.array(utility_scores), y=1-np.array(privacy_scores), s=70)

    plt.title('Utility/Privacy plot')
    plt.xlabel('Utility')
    plt.ylabel('Privacy')
    plt.grid(linestyle='--', linewidth=0.5)
    ax.set(xlim=(0, 1.03), ylim=(0, 1.03))
    plt.show()


if __name__ == '__main__':
    main()



