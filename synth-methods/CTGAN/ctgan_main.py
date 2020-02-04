#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import pandas as pd
import sys

try:
    from ctgan.synthesizer import CTGANSynthesizer
except ImportError as err:
    sys.exit("[ERROR] CTGAN library needs to be installed.\nError message: %s" % err)

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, "Base"))
from base import SynthesizerBase

# --- CTGAN main class
class SynthesizerCTGAN(SynthesizerBase):

    def __init__(self):
        self.num_samples_to_fit = None
        self.num_samples_to_synthesize = None
        self.discrete_column_names = None
        self.num_epochs = None
        self.random_state = None

        # instantiate SynthesizerBase from Base directory
        super().__init__()

    def fit_synthesizer(self, parameters_json_path, csv_path, metadata_json_path, verbose=True):
        """Fits CTGAN model and stores it internally
        Uses pre-stored data and metadata if use_stored_inputs=True (default).
        Stores the data and metadata within the class object if store_internally=True (default False).
        Returns the fitted model."""

        # Read data and metadata
        if verbose:
            print("\n[INFO] Reading input data and metadata from disk")
        input_data, metadata = self.read_data(csv_path, metadata_json_path, verbose)
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
        self.num_epochs = self.parameters['parameters']['num_epochs']
        self.random_state = self.parameters['parameters']['random_state']
        if verbose:
            print(f"\n[INFO] Reading CTGAN parameters from json file:\n"
                  f"num_samples_to_fit = {self.num_samples_to_fit}\n"
                  f"num_epochs = {self.num_epochs}\n"
                  f"random_state = {self.random_state}\n")

        # Extract discrete column names list from metadata
        # XXX NOTE: The list of discrete types needs to be updated when the format of the metadata is finalised
        # XXX Deal with DateTime in CTGAN
        self.discrete_column_names = [col['name'] for col in metadata['columns']
                                      if col['type'] in ['Categorical', 'Ordinal', 'DiscreteNumerical', "DateTime"]]

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

        # Instantiate CTGANSynthesizer object
        ctgan = CTGANSynthesizer()

        # Fit the model
        if verbose:
            print(f"\n[INFO] Fitting the CTGAN model, total number of epochs: {self.num_epochs}")
        ctgan.fit(data_sample, self.discrete_column_names, epochs=self.num_epochs)

        # Store the model
        self.model = ctgan

    def synthesize(self, output_path, num_samples_to_synthesize=False, store_internally=False, verbose=True):
        """Create synthetic data set using the fitted model. Stores the synthetic data
        within the class object if store_internally=True (default) and outputs the
        synthetic data to disk if output_filename is provided (default False)."""

        self.num_samples_to_synthesize = num_samples_to_synthesize
        # Synthesize the samples
        if not self.num_samples_to_synthesize:
            print(f"[INFO] number of samples to synthesize is set to {len(self.data_sample)}")
            self.num_samples_to_synthesize = len(self.data_sample)
        synthetic_data = self.model.sample(self.num_samples_to_synthesize)

        if verbose:
            print(f"\n[INFO] Synthesis: Created synthetic data set with the following characteristics:\n"
                  f"#rows:    {synthetic_data.shape[0]}\n#columns: {synthetic_data.shape[1]}")
            print(f"Column names:  {[col for col in synthetic_data.columns]}\n")

        # Write data to disk if needed
        if output_path:
            if not os.path.isdir(output_path):
                os.makedirs(output_path)
            if os.path.isfile(output_path):
                print(f"[WARNING] Output file {output_path} already exists and will be overwritten")
            synthetic_data.to_csv(os.path.join(output_path,"synthetic_data.csv"), index=False)

            with open(os.path.join(output_path,"ctgan_parameters.json"), 'w') as par:
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
    # Test if it works

    import uuid

    ctgan_syn = SynthesizerCTGAN()

    UUID_run = uuid.uuid1()

    output_path = "../../synth_output/ctgan_"+str(UUID_run)


    path2csv = os.path.join("../../datasets/generated/odi_nhs_ae/hospital_ae_data_deidentify.csv")
    path2meta = os.path.join("../../datasets/generated/odi_nhs_ae/hospital_ae_data_deidentify.json")
    path2params = os.path.join("tests", "parameters", "ctgan_parameters.json")

    ctgan_syn.fit_synthesizer(path2params, path2csv, path2meta)
    ctgan_syn.synthesize(num_samples_to_synthesize=200, output_path=output_path)



#    import ipdb; ipdb.set_trace()
