import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys

from ctgan import CTGANSynthesizer

# =========== INPUT
# >>>>>> CSV file
read_csv_file = False
path2csv = os.path.join("..", "..", "datasets", "generated", "odi_nhs_ae", "hospital_ae_data_deidentify.csv")
path2synth = "../../synth_output/ctgan_dbc77ed4-5252-11ea-9c88-a860b61fcc00/synthetic_data.csv"

# >>>>>> OBJ file
epochs2read = range(10, 180+10, 10)
num_CV_fold = 1
num_samples2generate = 9889

age2integer = {
        '0-17': 0,
        '18-24': 1,
        '25-44': 2,
        '45-64': 3,
        '65-84': 4,
        '85-': 5
    }

if read_csv_file:
    synth = pd.read_csv(path2synth)
    synth = synth.replace({"Age bracket": age2integer})
    synth_corr = synth.corr().iloc[3, 0]

    real = pd.read_csv(path2csv)
    real = real.replace({"Age bracket": age2integer})
    real_corr = real.corr().iloc[3, 0]

    print("Synthetic correlation is: ", synth_corr)
    print("Real correlation is: ", real_corr)
else:
    ctgan_model = CTGANSynthesizer()

    plt.figure()
    plt.subplot(2, 1, 1)
    for cvf in range(num_CV_fold):
        print(f"Cross Validation fold: {cvf}/{num_CV_fold}")
        loss_d = []
        loss_g = []
        corr2plot = []
        for myepoch in epochs2read:
            ctgan_model.load("checkpoints/model_%i.obj" % myepoch)
            loss_d.append(ctgan_model.loss_d)
            loss_g.append(ctgan_model.loss_g)
            synth = ctgan_model.sample(num_samples2generate)
            synth = synth.replace({"Age bracket": age2integer})
            corr2plot.append(synth[['Time in A&E (mins)', 'Age bracket']].astype(float).corr().iloc[0, 1]*100.)
        plt.plot(epochs2read, corr2plot, c='k', ls='-', marker="o", alpha=1.0)

    plt.xlabel("#epochs", size=24)
    plt.ylabel("Correlation", size=24)
    plt.xticks(size=18)
    plt.yticks(size=18)
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.plot(epochs2read, loss_d, c='k', ls='-', marker="o")
    plt.xlabel("#epochs", size=16)
    plt.ylabel("Loss Discriminator", size=16)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.plot(epochs2read, loss_g, c='k', ls='-', marker="o")
    plt.xlabel("#epochs", size=16)
    plt.ylabel("Loss Generator", size=16)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.grid()

    plt.tight_layout()
    plt.show()
