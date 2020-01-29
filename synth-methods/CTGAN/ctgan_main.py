import json
import os
import pandas as pd
import sys
from ctgan.synthesizer import CTGANSynthesizer
sys.path.append("../Base")
from base import SynthesizerBase


class SynthesizerCTGAN(SynthesizerBase):

    def __init__(self):
        self.num_samples_for_fitting = None
        self.num_epochs = None
        self.num_samples_to_synthesise = None
        self.random_state = None
        self.column_names = None
        super().__init__()

    def _read_data(self, csv_path, json_path, synthesis_name, store_internally=False, verbose=True):
        """This is intended as a place to to call the parent _read_data(...) and also add method-specific
        data pre-processing after reading from disk. If no pre-processing is required then it can just
        call the parent _read_data(...)"""
        return super()._read_data(csv_path, json_path, synthesis_name,
                                  store_internally=False, verbose=True)

    def fit_synthesizer(self, csv_path, metadata_json_path, synthesis_name, parameters_json_path,
                        store_internally=False, use_stored_inputs=False, verbose=True):
        """Reads input data and metadata, fits CTGAN model and stores it internally.
        Stores the data and metadata within the class object if store_internally=True (default False).
        Uses pre-stored data and metadata if use_stored_inputs=True (default False).
        Returns the fitted model."""

        # Read data and metadata
        if use_stored_inputs:
            if self.input_data is None:
                sys.exit("\nNo input data file is stored in the object.")
            elif self.metadata is None:
                sys.exit("\nNo metadata file is stored in the object.")
            else:
                if verbose:
                    print("\n[INFO] Reading data and metadata previously stored within class object")
                input_data = self.input_data
                metadata = self.metadata
        else:
            if verbose:
                print("\n[INFO] Reading input data and metadata from disk")
            input_data, metadata = self._read_data(csv_path, metadata_json_path,
                                                   synthesis_name, store_internally, verbose)

        if verbose:
            print("\nSUMMARY")
            print("\nDataframe dimensions\n#rows:    {}\n#columns: {}".format(input_data.shape[0],
                                                                              input_data.shape[1]))
            print("\nColumn name \t Type\n" + 11 * "--")
            for col in metadata['columns']:
                print("{} \t {}".format(col['name'], col['type']))

        # Extract parameters from json parameter file
        with open(parameters_json_path) as parameters_json:
            self.parameters = json.load(parameters_json)
        self.num_samples_for_fitting = self.parameters['parameters'][0]['value']
        self.num_epochs = self.parameters['parameters'][1]['value']
        self.random_state = self.parameters['parameters'][3]['value']
        if verbose:
            print(f"\n[INFO] Reading CTGAN parameters from json file:\n"
                  f"num_samples_for_fitting = {self.num_samples_for_fitting}\n"
                  f"num_epochs = {self.num_epochs}\n"
                  f"random_state = {self.random_state}\n")

        # Extract column names list from metadata
        self.column_names = [col['name'] for col in metadata['columns']]

        # Draw random sample from input data with requested size
        if self.num_samples_for_fitting == 0:
            sys.exit('\nNumber of samples for fitting cannot be 0')
        elif self.num_samples_for_fitting != -1:
            if verbose:
                print(f"\n[INFO] Sampling {self.num_samples_for_fitting} rows from input data")
            data_sample = input_data.sample(n=self.num_samples_for_fitting,
                                            random_state=self.random_state)
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
        ctgan.fit(data_sample, self.column_names, epochs=self.num_epochs)

        # Store the model
        self.model = ctgan

        return ctgan

    def synthesize(self, num_samples_to_synthesize=False, store_internally=False,
                   output_filename=False, verbose=True):
        """Creates synthetic data set using the fitted model. Stores the synthetic data
        within the class object if store_internally=True (default False) and outputs the
        synthetic data to disk if output_filename is provided (default False).
        Returns the synthetic data frame."""

        # Synthesize the samples
        if num_samples_to_synthesize:
            synthetic_data = self.model.sample(num_samples_to_synthesize)
        else:
            synthetic_data = self.model.sample(self.parameters['parameters'][2]['value'])
        if verbose:
            print(f"\n[INFO] Synthesis: Created synthetic data set with the following characteristics:\n"
                  f"#rows:    {synthetic_data.shape[0]}\n#columns: {synthetic_data.shape[1]}")
            print(f"Column names:  {[col for col in synthetic_data.columns]}\n")

        # Store internally if needed
        if store_internally:
            self.synthetic_data = synthetic_data
            if verbose:
                print("[INFO] Stored synthesised dataset internally")

        # Write data to disk if needed
        if output_filename:
            output_path = f'../../synthetic-output/{self.synthesis_name}/{self.dataset_name}/CTGAN/'
            output_filepath = output_path + output_filename
            if not os.path.isdir(output_path):
                os.makedirs(output_path)
            if os.path.isfile(output_filepath):
                print(f"[WARNING] Output file {output_filepath} already exists and will be overwritten")
            synthetic_data.to_csv(output_filepath)
            if verbose:
                print(f"[INFO] Stored synthesized dataset to file: {output_filepath}")

        return synthetic_data

# Test if it works
syn = SynthesizerCTGAN()
syn.fit_synthesizer("tests/data/test_CTGAN_io.csv", "tests/data/test_CTGAN_io_data.json",
                  "Synthesis00", "tests/parameters/ctgan_parameters.json", True, False, True)
output = syn.synthesize(150, True, "test.csv", True)
print("Output df head:\n", output.head())
#print(syn.model)
#print(syn.parameters)
