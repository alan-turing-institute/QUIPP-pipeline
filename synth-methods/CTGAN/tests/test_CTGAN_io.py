import json
import pandas as pd
from ctgan.synthesizer import CTGANSynthesizer

from utils import read_csv, synthesizer_fit, generate_samples


inp_csv_path="./tests/data/test_CTGAN_io.csv"
inp_metadata_json_path="./tests/data/test_CTGAN_io_data.json" 
inp_num_epochs=20
inp_random_state=10
inp_num_samples_from_data=False
inp_metadata=False
inp_num_syn_samples_to_generate=100

out_batchsize = 500
out_embed_dim = 128

def test_CTGAN_read_csv():
    data, discrete_columns, metadata = read_csv(inp_csv_path, inp_metadata_json_path) 
    assert "columns" in metadata, "columns' is not in %s" % inp_metadata_json_path
    assert len(metadata['columns']) > 0, "Number of columns is <= 0"
    assert "name" in metadata['columns'][0], "'name' is not in %s" % inp_metadata_json_path

def test_CTGAN_synthesizer_fit():
    data, discrete_columns, metadata = read_csv(inp_csv_path, inp_metadata_json_path)
    ctgan = synthesizer_fit(data, discrete_columns, metadata=metadata, 
                            num_epochs=inp_num_epochs, random_state=inp_random_state, 
                            num_samples_from_data=inp_num_samples_from_data)
    assert ctgan.batch_size == out_batchsize, "Batch size should be %s" % out_batchsize
    assert ctgan.embedding_dim == out_embed_dim, "Embedding dim should be %s" % out_embed_dim


def test_CTGAN_generate_sample():
    data, discrete_columns, metadata = read_csv(inp_csv_path, inp_metadata_json_path)
    ctgan = synthesizer_fit(data, discrete_columns, metadata=metadata, 
                            num_epochs=inp_num_epochs, random_state=inp_random_state, 
                            num_samples_from_data=inp_num_samples_from_data)
    samples = generate_samples(ctgan, inp_num_syn_samples_to_generate)
    assert len(samples) == inp_num_syn_samples_to_generate, "number of generated synthetic samples is not equal to the requested one."
