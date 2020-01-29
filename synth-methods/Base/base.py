import json
import os
import pandas as pd
import sys
import ntpath


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


class SynthesizerBase:

    def __init__(self):
        self.dataset_name = None
        self.synthesis_name = None
        self.input_data = None
        self.metadata = None
        self.parameters = None
        self.model = None
        self.synthetic_data = None

    def read_data(self, csv_path, json_path, synthesis_name, store_internally=False, verbose=True):
        """Reads input data from .csv file and metadata from .json file.
        Stores the data and metadata within the class object if
        store_internally=True (default False). User has to define a synthesis name (synth_name)
        Returns data (pandas dataframe) and metadata (dictionary)"""

        if not os.path.isfile(csv_path):
            sys.exit("File does not exist: %s" % csv_path)

        if not os.path.isfile(json_path):
            sys.exit("File does not exist: %s" % json_path)

        # Extract csv filename only, remove extension and use as dataset_name
        # synthesis_name is provided by user
        self.dataset_name = path_leaf(csv_path)[:-4]
        self.synthesis_name = synthesis_name

        data = pd.read_csv(csv_path)

        with open(json_path) as metadata_json:
            metadata = json.load(metadata_json)

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

        if store_internally:
            self.input_data = data
            self.metadata = metadata
            if verbose:
                print("\n[INFO] Stored input data and metadata internally")

        return data, metadata

    def fit_synthesizer(self):
        pass

    def synthesize(self):
        pass
