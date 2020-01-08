import numpy as np
import sys
from ctgan import load_demo
from ctgan.data import read_csv
from ctgan.synthesizer import CTGANSynthesizer

def test_CTGAN_pipeline():
    num_samples_from_data = 5000
    num_syn_samples_to_generate = 1000   
    random_state = 10
    num_epochs = 20

    data = load_demo()
    discrete_columns = [
        'workclass',
        'education',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'native-country',
        'income'
    ]

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

    samples = ctgan.sample(num_syn_samples_to_generate)

    print(samples.head())

    assert(False)
