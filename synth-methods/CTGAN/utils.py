import json
import os
import pandas as pd
import sys
from ctgan.synthesizer import CTGANSynthesizer

def read_csv(csv_path, metadata_json_path):

    if not os.path.isfile(csv_path):
        sys.exit("File does not exist: %s" % csv_path)

    if not os.path.isfile(metadata_json_path):
        sys.exit("File does not exist: %s" % metadata_json_path)

    data = pd.read_csv(csv_path) 

    discrete_columns = []
    with open(metadata_json_path) as metadata_json:
        metadata = json.load(metadata_json)
    
    for one_column in metadata['columns']:
        if one_column['type'] in ['categorical']:
            discrete_columns.append(one_column['name'])
    
    return data, discrete_columns, metadata

def synthesizer_fit(data, discrete_columns, metadata=False, 
                    num_epochs=20, random_state=10, num_samples_from_data=False):
    
    if num_samples_from_data:
        print(f"[INFO] Sample {num_samples_from_data} from data!")
        data_sample = data.sample(n=num_samples_from_data, 
                                    random_state=random_state)
    else:
        data_sample = data

    print(data_sample.describe())
    print(discrete_columns)
    
    ctgan = CTGANSynthesizer()
    print(ctgan)

    print(f"[INFO] Start fitting the model, total number of epochs: {num_epochs}")
    ctgan.fit(data_sample, discrete_columns, epochs=num_epochs)

    return ctgan

def generate_samples(ctgan, num_syn_samples_to_generate=100):
    samples = ctgan.sample(num_syn_samples_to_generate)
    print(samples.head())
    return samples



