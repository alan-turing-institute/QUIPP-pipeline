import json
import sys
import os
import subprocess
import shutil
import seaborn as sns
import matplotlib.pyplot as plt

in_path = "parameter_files/generated-synthpop-dr/"
out_path = "../run-inputs/"
synth_path = "../synth-output/"


def copy_parameter_files():
    f_list = os.listdir(path=in_path)
    for f in f_list:
        shutil.copyfile(os.path.join(in_path, f), os.path.join(out_path, f))


def run_pipeline():
    cmd = '''
    cd ..
    make   
    '''
    subprocess.check_output(cmd, shell=True)


def main():

    copy_parameter_files()
    #run_pipeline()

    f_list = os.listdir(path=in_path)
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

    print(privacy_scores)
    print(utility_scores)

    sns.set()
    cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
    synthesised_columns = [0, 2, 4, 6]
    ax = sns.scatterplot(x=utility_scores, y=privacy_scores, hue=synthesised_columns, palette=cmap)
    plt.title('Utility/Privacy/#Synthesised columns plot')
    plt.xlabel('Utility (Mean abs. relative difference in F1)')
    plt.ylabel('Expected Match Risk')
    ax.legend(loc='upper right', frameon=False, title="# Synthesised columns")
    plt.show()


if __name__ == '__main__':
    main()



