import os
import pandas as pd
import pytest
import sys
from ctgan.synthesizer import CTGANSynthesizer as ctgan_original_model_class

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
from ctgan_main import SynthesizerCTGAN

# Fixed inputs for tests
path2csv = os.path.join("tests", "data", "test_CTGAN_io.csv")
path2meta = os.path.join("tests", "data", "test_CTGAN_io_data.json") 
path2params = os.path.join("tests", "parameters", "ctgan_parameters.json")
dataset_name = 'test_CTGAN_io'

output_path = "./synthetic-output/dataset-name/test.csv"

inp_discrete_columns = ['id', 'age', 'origin', 'favourite_food']
dis_dim = (256, 256)
gen_dim = (256, 256)
embed_dim = 128
batch_size = 500
num_epochs = 20
random_state = 1234

num_samples_to_synthesize = 200
num_samples_to_fit = -1

@pytest.fixture
def ctgan_syn(): 
    ctgan_syn = SynthesizerCTGAN()
    return ctgan_syn

def test_SynthesizerCTGAN_init(ctgan_syn):
    assert ctgan_syn.discrete_column_names == None, "discrete_column_names is not instantiated!"
def test_SynthesizerCTGAN_fit_synthesizer(ctgan_syn):
    ctgan_syn.fit_synthesizer(path2params, path2csv, path2meta)
    assert isinstance(ctgan_syn.model, ctgan_original_model_class)
    assert ctgan_syn.model.dis_dim == dis_dim, "Unexpected discriminative dimensions %s" % dis_dim
    assert ctgan_syn.model.gen_dim == gen_dim, "Unexpected generatove dimensions %s" % gen_dim
    assert ctgan_syn.model.embedding_dim == embed_dim, "Unexpected embedding dimension: %s" % embed_dim
    assert ctgan_syn.discrete_column_names == inp_discrete_columns, "Discrete columns do not match!"
    assert ctgan_syn.num_epochs == num_epochs, "Unexpected number of epochs %s" % num_epochs 
    assert ctgan_syn.random_state == random_state, "Unexpected random state %s" %  random_state 
    assert ctgan_syn.num_samples_to_fit == num_samples_to_fit, "Unexpected number of samples to fit %s" % num_samples_to_fit 
    assert ctgan_syn.model.batch_size == batch_size, "Unexpected batch size: %s" % batch_size

def test_SynthesizerCTGAN_synthesize(ctgan_syn):
    ctgan_syn.fit_synthesizer(path2params, path2csv, path2meta)
    ctgan_syn.synthesize(output_path=output_path, num_samples_to_synthesize=num_samples_to_synthesize, store_internally=True)
    assert ctgan_syn.dataset_name == dataset_name, "Unexpected dataset name: %s" % dataset_name 
    assert ctgan_syn.num_samples_to_synthesize == num_samples_to_synthesize, "Unexpected num_samples_to_synthesize: %s" %num_samples_to_synthesize
    assert os.path.isfile(output_path), "File %s is not created!" % output_path
    assert os.path.isdir("./synthetic-output/dataset-name"), "Directory is not created"
    assert os.path.isdir("./synthetic-output"), "Directory is not created"
    read_csv_file = pd.read_csv(output_path) 
    assert len(read_csv_file) ==  num_samples_to_synthesize, "Number of rows in the generated CSV file is not equal to num_samples_to_synthesize: %s" % num_samples_to_synthesize