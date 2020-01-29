import json
import os
import pandas as pd
import sys


class SynthesizerBase:

    def __init__(self):
        self.input_data = None
        self.metadata = None

    def read_data(self, csv_path, json_path, store_internally=False):
        """Reads input data from .csv file and metadata from .json file.
        Stores the data and metadata within the class object if
        store_internally=True (default False). Returns data (pandas dataframe)
        and metadata (dictionary)"""

        if not os.path.isfile(csv_path):
            sys.exit("File does not exist: %s" % csv_path)

        if not os.path.isfile(json_path):
            sys.exit("File does not exist: %s" % json_path)

        data = pd.read_csv(csv_path)

        with open(json_path) as metadata_json:
            metadata = json.load(metadata_json)

        if store_internally:
            self.input_data = data
            self.metadata = metadata

        if not ('columns' in metadata):
            sys.exit("Metadata file does not contain 'columns' key. Please refer to the documentation.")

        metadata_cols_number = len(metadata['columns'])
        if not metadata_cols_number == data.shape[1]:
            sys.exit("Number of columns mismatch between data ({}) and metadata ({})".format(metadata_cols_number,
                                                                                             data.shape[1]))

        for index, name_data in enumerate(data.columns):
            name_metadata = metadata['columns'][index]['name']
            if not (name_data == name_metadata):
                print("\nWARNING: Column name mismatch between data and metadata: {} {}".format(name_data, name_metadata))

        print("\nSUMMARY")
        print("\nDataframe #rows/#columns: {}/{}".format(data.shape[0],data.shape[1]))
        print("Column name \t Type")
        for col in metadata['columns']:
            print("{} \t {}".format(col['name'], col['type']))

        return data, metadata
