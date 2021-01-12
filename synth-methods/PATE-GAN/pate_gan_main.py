#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import pandas as pd
import sys

try:
    # from pate_gan_source import PateGanSynthesizer
    from pate_gan import PateGan as PateGanSynthesizer
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), 
                                 os.path.pardir, 
                                 os.path.pardir, 
                                 "libs", "synthetic_data_release", "generative_models"))
    sys.path.append(os.path.join(os.path.dirname(__file__), 
                                 os.path.pardir, 
                                 os.path.pardir, 
                                 "libs", "synthetic_data_release"))
    from pate_gan import PateGan as PateGanSynthesizer
except:
    err_msg = "[ERROR] could not import PateGanSynthesizer.\n"
    err_msg += "Refer to the README file to set up PATE-GAN"
    sys.exit(err_msg)

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, "Base"))
from base import SynthesizerBase

# --- PATE-GAN main class
class SynthesizerPateGAN(SynthesizerBase):

    def __init__(self):
        self.num_samples_to_fit = None
        self.num_samples_to_synthesize = None
        self.discrete_column_names = None
        self.num_epochs = None
        self.random_state = None

        # instantiate SynthesizerBase from Base directory
        super().__init__()

    def fit_synthesizer(self, parameters_json_path, csv_path, metadata_json_path, verbose=True):
        """
        Fits PATE-GAN model and stores the model internally

        Arguments:
            parameters_json_path: path to a json file which contains PATE-GAN parameters (user inputs)
            csv_path: real data
            metadata_json_path: file describing the CSV file
        """

        # Read data and metadata
        if verbose:
            print("\n[INFO] Reading input data and metadata from disk")
        input_data, metadata = self.read_data(csv_path, metadata_json_path, verbose)
        print(f"[INFO] #rows before removing NaN: {len(input_data)}")
        input_data.dropna(inplace=True)
        print(f"[INFO] #rows after removing NaN: {len(input_data)}")
        with open(metadata_json_path) as metadata_json:
            self.metadata = json.load(metadata_json)

        if verbose:
            print("\nSUMMARY")
            print("-------")
            print("Dataframe dimensions\n#rows:    {}\n#columns: {}".format(input_data.shape[0],
                                                                            input_data.shape[1]))
            print("\nColumn name \t Type\n" + 11 * "--")
            for col in metadata['columns']:
                print("{} \t {}".format(col['name'], col['type']))

        # Extract parameters from json parameter file
        with open(parameters_json_path) as parameters_json:
            self.parameters = json.load(parameters_json)
        self.num_samples_to_fit = self.parameters['parameters']['num_samples_to_fit']
        self.random_state = self.parameters['parameters']['random_state']
        self.num_datasets_to_synthesize = self.parameters["parameters"]["num_datasets_to_synthesize"]
        self.num_samples_to_synthesize = self.parameters["parameters"]["num_samples_to_synthesize"]
        self.eps = self.parameters['parameters']['epsilon']
        self.delta = self.parameters['parameters']['delta']
        if verbose:
            print(f"\n[INFO] Reading PATE-GAN parameters from json file:\n"
                  f"num_samples_to_fit = {self.num_samples_to_fit}\n"
                  f"random_state = {self.random_state}\n")

        # Extract discrete column names list from metadata
        self.metadata['categorical_columns'] = []
        self.metadata['ordinal_columns'] = []
        self.metadata['continuous_columns'] = []
        for i, col in enumerate(metadata['columns']):
            if col['type'] in ['Categorical', 'DiscreteNumerical', "DateTime"]:
                self.metadata['categorical_columns'].append(i)
            elif col['type'] in ['Ordinal']:
                self.metadata['ordinal_columns'].append(i)
            else:
                self.metadata['continuous_columns'].append(i)

        categorical_columns = sorted(self.metadata['categorical_columns'] + self.metadata['ordinal_columns'])

        for one_col in categorical_columns:
            unique_values = input_data[metadata['columns'][one_col]['name']].unique()
            unique_values.sort()
            self.metadata['columns'][one_col]['size'] = len(unique_values)
            self.metadata['columns'][one_col]['i2s'] = unique_values.tolist()

        for one_col in self.metadata['continuous_columns']:
            self.metadata['columns'][one_col]['min'] = input_data[metadata['columns'][one_col]['name']].min()
            self.metadata['columns'][one_col]['max'] = input_data[metadata['columns'][one_col]['name']].max()

        # Draw random sample from input data with requested size
        if self.num_samples_to_fit == 0:
            sys.exit('\nNumber of samples for fitting cannot be 0')
        elif self.num_samples_to_fit != -1:
            if verbose:
                print(f"\n[INFO] Sampling {self.num_samples_to_fit} rows from input data")
            data_sample = input_data.sample(n=self.num_samples_to_fit, random_state=self.random_state)
        else:
            data_sample = input_data

        if verbose:
            print("[INFO] Summary of the data frame that will be used for fitting:")
            print(data_sample.describe())

        # Instantiate PateGanSynthesizer object
        pategan = PateGanSynthesizer(self.metadata, self.eps, self.delta)

        # Fit the model
        if verbose:
            print(f"\n[INFO] Fitting the PATE-GAN model.")
        self.num_data_sample = len(data_sample)
        pategan.fit(data_sample)

        # Store the model
        self.model = pategan

    def synthesize(self, output_path, num_samples_to_synthesize=-1, store_internally=False, verbose=True):
        """Create synthetic data set using the fitted model. Stores the synthetic data
        within the class object if store_internally=True (default) and outputs the
        synthetic data to disk if output_filename is provided (default False)."""

        if self.num_samples_to_synthesize == None:
            self.num_samples_to_synthesize = num_samples_to_synthesize
        # Synthesize the samples
        if self.num_samples_to_synthesize < 0:
            print(f"[INFO] number of samples to synthesize is set to {self.num_data_sample}")
            self.num_samples_to_synthesize = self.num_data_sample

        # Write data to disk if needed
        if output_path:
            if not os.path.isdir(output_path):
                os.makedirs(output_path)
            for i_syn in range(1, self.num_datasets_to_synthesize + 1):
                synthetic_data = self.model.generate_samples(self.num_samples_to_synthesize)

                if verbose:
                    print(f"\n[INFO] Synthesis: Created synthetic data set with the following characteristics:\n"
                          f"#rows:    {synthetic_data.shape[0]}\n#columns: {synthetic_data.shape[1]}")
                    print(f"Column names:  {[col for col in synthetic_data.columns]}\n")

                synthetic_data.to_csv(os.path.join(output_path, f"synthetic_data_{i_syn}.csv"), index=False)

            with open(os.path.join(output_path,"pategan_parameters.json"), 'w') as par:
                json.dump(self.parameters, par)

            with open(os.path.join(output_path,"data_description.json"), 'w') as meta:
                json.dump(self.metadata, meta)

            if verbose:
                print(f"[INFO] Stored synthesized dataset to file: {output_path}")

        # Store internally if needed
        if store_internally:
            self.synthetic_data = synthetic_data
            if verbose:
                print("[INFO] Stored synthesized dataset internally")

if __name__ == "__main__":
    print("[WARNING] run PATE-GAN via SynthesizerPateGAN class.")