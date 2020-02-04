#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import pandas as pd
import re
import sys

# --- Base class to be used in synth-methods 
class SynthesizerBase:

    def __init__(self):
        self.dataset_name = None
        self.parameters = None
        self.metadata = None
        self.model = None

    def read_data(self, csv_path, json_path, verbose=True):
        """Reads input data from .csv file and metadata from .json file.
        Stores the data and metadata within the class object if
        store_internally=True (default). User can define a synthesis name (synth_name)
        Returns data (pandas dataframe) and metadata (dictionary)"""

        # --- CHECK csv and json files
        if not os.path.isfile(csv_path):
            sys.exit("File does not exist: %s" % csv_path)
        if not os.path.isfile(json_path):
            sys.exit("File does not exist: %s" % json_path)

        # Read csv from file
        data = pd.read_csv(csv_path)
        # Extract csv filename only, remove extension and use as dataset_name
        self.dataset_name = re.split(".csv", os.path.basename(csv_path), flags=re.IGNORECASE)[0]

        # Read json from file
        with open(json_path) as metadata_json:
            metadata = json.load(metadata_json)

        # >>> CHECK JSON file
        if not ('columns' in metadata):
            sys.exit("\nMetadata file does not contain 'columns' key. Please refer to the documentation.")

        metadata_cols_number = len(metadata['columns'])
        if not metadata_cols_number == data.shape[1]:
            sys.exit("\nNumber of columns mismatch between data ({}) and metadata ({})".format(metadata_cols_number,
                                                                                               data.shape[1]))

        for index, name_data in enumerate(data.columns):
            name_metadata = metadata['columns'][index]['name']
            if not (name_data == name_metadata):
                print("\n[WARNING]: Column name mismatch between data and metadata: {} {}".format(name_data,
                                                                                                  name_metadata))
        # <<<

        return data, metadata

    def fit_synthesizer(self):
        """Leave empty. Method-specific implementation should be provided in child class of Base"""
        pass

    def synthesize(self):
        """Leave empty. Method-specific implementation should be provided in child class of Base"""
        pass
