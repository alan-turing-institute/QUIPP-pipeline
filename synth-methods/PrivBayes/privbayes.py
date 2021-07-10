import json
import os
import numpy as np
import pandas as pd
import random
import sys

from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.ModelInspector import ModelInspector
from DataSynthesizer.lib.utils import read_json_file, display_bayesian_network

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, "Base"))
from base import SynthesizerBase


class SynthesizerPrivBayes(SynthesizerBase):
    """Implements dataset synthesis with PrivBayes."""

    def __init__(self, parameters_json_path):
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
        self.preconfigured_bn = None
        self.user_pool = 1

        # instantiate SynthesizerBase from Base directory
        super().__init__()

        # initialize parameters
        with open(parameters_json_path) as parameters_json:
            self.parameters = json.load(parameters_json)
        self.category_threshold = self.parameters['parameters']['category_threshold']
        self.epsilon = self.parameters['parameters']['epsilon']
        self.k = self.parameters['parameters']['k']
        self.keys = self.parameters['parameters']['keys']
        self.histogram_bins = self.parameters['parameters']['histogram_bins']
        self.random_state = self.parameters['parameters']['random_state']
        self.preconfigured_bn = self.parameters['parameters']['preconfigured_bn']
        if self.parameters['parameters'].get('user_pool'):
            self.user_pool = self.parameters['parameters']['user_pool']

        self.num_samples_to_synthesize = self.parameters['parameters']['num_samples_to_synthesize']
        
        self.save_description = self.parameters['parameters'].get('save_description')
        if self.save_description is None:
            self.save_description = True

    def fit_synthesizer(self, csv_path, metadata_json_path,
                        output_path, verbose=True):
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
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        np.random.default_rng(self.random_state)
        if verbose:
            print(f"[INFO] Reading PrivBayes parameters from json file:\n"                  
                  f"category_threshold = {self.category_threshold}\n"    
                  f"epsilon = {self.epsilon}\n"   
                  f"k = {self.k}\n" 
                  f"keys = {self.keys}\n" 
                  f"histogram_bins = {self.histogram_bins}\n" 
                  f"random_state = {self.random_state}\n"
                  f"preconfigured_bn = {self.preconfigured_bn}\n")

        # Convert datatype metadata in .json file to datatype dictionaries usable by PrivBayes
        float_dict = {col['name']: 'Float' for col in self.metadata['columns']
                      if col['type'] == 'ContinuousNumerical'}

        # for integer columns, make sure the ones classified as categorical by the .json file are
        # included in the int_dict used by PrivBayes
        df = pd.read_csv(csv_path)
        integer_types = ['int_', 'intp', 'int8', 'int16', 'int32', 'int64']
        float_types = ['float_', 'float16', 'float32', 'float64']
        integer_columns = list(df.select_dtypes(include=integer_types).columns)
        float_columns = list(df.select_dtypes(include=float_types).columns)

        int_dict = {col['name']: 'Integer' for col in self.metadata['columns']
                    if (col['type'] == 'DiscreteNumerical') or (col['type'] in ['Ordinal', 'Categorical'] and
                                                                col['name'] in integer_columns)}

        str_dict = {col['name']: 'String' for col in self.metadata['columns']
                    if col['type'] in ['Ordinal', 'Categorical', 'String']
                    and col['name'] not in integer_columns
                    and col['name'] not in float_columns}

        dt_dict = {col['name']: 'DateTime' for col in self.metadata['columns']
                   if col['type'] == 'DateTime'}
        # Combine all datatypes into one dictionary
        self.datatypes = {**float_dict, **int_dict, **str_dict, **dt_dict}

        # Add all categorical variables in a dict
        self.categorical_attributes = {col['name']: True for col in self.metadata['columns']
                                       if col['type'] in ['Categorical', 'Ordinal']}

        # Add all keys in a dict
        self.candidate_keys = {}
        for col in self.metadata['columns']:
            if col['name'] in self.keys:
                self.candidate_keys[col['name']] = True
            else:
                self.candidate_keys[col['name']] = False

        # Instantiate describer object, describe the dataset and get probabilities
        if verbose:
            print(f"[INFO] Getting variable descriptions and Bayesian Network structure with PrivBayes")
        self.describer = DataDescriber(category_threshold=self.category_threshold,
                                       histogram_bins=self.histogram_bins)

        # Train the Bayesian network
        import ipdb; ipdb.set_trace()
        self.describer.describe_dataset_in_correlated_attribute_mode(dataset_file=csv_path,
                                                                     epsilon=self.epsilon,
                                                                     k=self.k,
                                                                     attribute_to_datatype=self.datatypes,
                                                                     attribute_to_is_categorical=self.categorical_attributes,
                                                                     attribute_to_is_candidate_key=self.candidate_keys,
                                                                     seed=self.random_state,
                                                                     bayesian_network=self.preconfigured_bn,
                                                                     user_pool=self.user_pool
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
        # Synthesize the samples
        generator = DataGenerator()
        generator.generate_dataset_in_correlated_attribute_mode(self.num_samples_to_synthesize,
                                                                self.description_file)

        # save synthetic data to disk
        generator.save_synthetic_data(os.path.join(output_path, "synthetic_data_1.csv"))

    def __del__(self):
        if not self.save_description:
            os.remove(self.description_file)
