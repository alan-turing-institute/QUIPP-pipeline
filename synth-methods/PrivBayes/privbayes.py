import json
import os
import pandas as pd
import sys

from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.ModelInspector import ModelInspector
from DataSynthesizer.lib.utils import read_json_file, display_bayesian_network

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, "Base"))
from base import SynthesizerBase


class SynthesizerPrivBayes(SynthesizerBase):
    """Implements dataset synthesis with PrivBayes."""

    def __init__(self):
        self.metadata = None
        self.parameters = None
        self.num_samples_to_synthesize = None
        self.category_threshold = 10
        self.epsilon = None
        self.k = None
        self.categorical_attributes = None
        self.datatypes = None
        self.keys = None
        self.histogram_bins = None
        self.random_state = None
        self.describer = None
        self.description_file = None

        # instantiate SynthesizerBase from Base directory
        super().__init__()

    def fit_synthesizer(self, parameters_json_path, csv_path,
                        metadata_json_path, output_path,
                        verbose=True):
        """
        Wrapper around PrivBayes' DataDescriber. Finds data types and
        conditional probabilities, creates a description file.

        Parameters
        ----------
        parameters_json_path : string
            Path to a .json file which contains PrivBayes parameters (user inputs)
        csv_path: string
            Path to the original dataset
        metadata_json_path: string
            Path to the .json file describing the original dataset
        output_path: string
            Path where output description file will be written
        verbose: bool
            Whether to print all information/warnings output or not
        """

        with open(metadata_json_path) as metadata_json:
            self.metadata = json.load(metadata_json)

        # Extract parameters from json parameter file
        with open(parameters_json_path) as parameters_json:
            self.parameters = json.load(parameters_json)
        self.category_threshold = self.parameters['parameters']['category_threshold']
        self.epsilon = self.parameters['parameters']['epsilon']
        self.k = self.parameters['parameters']['k']
        self.keys = self.parameters['parameters']['keys']
        self.histogram_bins = self.parameters['parameters']['histogram_bins']
        self.random_state = self.parameters['parameters']['random_state']
        if verbose:
            print(f"[INFO] Reading PrivBayes parameters from json file:\n"                  
                  f"category_threshold = {self.category_threshold}\n"    
                  f"epsilon = {self.epsilon}\n"   
                  f"k = {self.k}\n" 
                  f"histogram_bins = {self.histogram_bins}\n" 
                  f"random_state = {self.random_state}\n")

        # Convert datatype metadata in .json file to datatype dictionary usable by PrivBayes
        float_dict = {col['name']: 'Float' for col in self.metadata['columns']
                      if col['type'] == 'ContinuousNumerical'}
        int_dict = {col['name']: 'Integer' for col in self.metadata['columns']
                    if col['type'] in ['DiscreteNumerical', 'CategoricalNumerical']}
        str_dict = {col['name']: 'String' for col in self.metadata['columns']
                    if col['type'] in ['Ordinal', 'Categorical']}
        # QUIPP DateTime types are converted to PrivBayes String because
        # conversion to PrivBayes DateTime (or leaving PrivBayes to automatically assign)
        # throws an error
        dt_dict = {col['name']: 'String' for col in self.metadata['columns']
                   if col['type'] == 'DateTime'}
        # Combine all datatypes into one dictionary
        self.datatypes = {**float_dict, **int_dict, **str_dict, **dt_dict}

        # Add all categorical variables in a dict - DateTime is treated as categorical
        self.categorical_attributes = {col['name']: True for col in self.metadata['columns']
                                       if col['type'] in ['Categorical', 'CategoricalNumerical', 'Ordinal', 'DateTime']}

        # Instantiate describer object, describe the dataset and get probabilities
        if verbose:
            print(f"[INFO] Getting variable descriptions and Bayesian Network structure with PrivBayes")
        self.describer = DataDescriber(category_threshold=self.category_threshold,
                                       histogram_bins=self.histogram_bins)

        # Train the Bayesian network
        self.describer.describe_dataset_in_correlated_attribute_mode(dataset_file=csv_path,
                                                                     epsilon=self.epsilon,
                                                                     k=self.k,
                                                                     attribute_to_datatype=self.datatypes,
                                                                     attribute_to_is_categorical=self.categorical_attributes,
                                                                     attribute_to_is_candidate_key=self.keys,
                                                                     seed=self.random_state
                                                                     )

        # write and print output
        self.description_file = os.path.join(output_path, "description.json")
        self.describer.save_dataset_description_to_file(self.description_file)
        display_bayesian_network(self.describer.bayesian_network)

    def synthesize(self, output_path):
        """Wrapper around PrivBayes' correlated attribute synthesis. Creates synthetic data set
        using a Bayesian Network that has been trained by fit_synthesizer().

        Parameters
        ----------
        output_path : string
            Path where output synthetic .csv will be written.
        """

        # how many samples to synthesize
        self.num_samples_to_synthesize = self.parameters['parameters']['num_samples_to_synthesize']

        # Synthesize the samples
        generator = DataGenerator()
        generator.generate_dataset_in_correlated_attribute_mode(self.num_samples_to_synthesize,
                                                                self.description_file)

        # save synthetic data to disk
        generator.save_synthetic_data(os.path.join(output_path, "synthetic_data_1.csv"))
