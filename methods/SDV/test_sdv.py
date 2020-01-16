import sys
import pytest
import pandas as pd
from util import get_shape, read_data

@pytest.fixture
def df():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'

    columns = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'income']

    df = pd.read_csv(url, names=columns)
    return df

def test_get_shape(df):
    assert get_shape(df) == (32561, 15)

def test_read_data():
    df = read_data('adults')
    assert df.shape == (32561, 15)
    assert df['age'].min() > 16
    assert df['age'].max() < 100
    assert len(df['sex'].unique()) == 2
    assert df.columns[2] == 'fnlwgt'

def test_pipeline():
    df = read_data('ons')

    metadata = {
        "tables": [
            {
                "fields": [
                    {
                        "name": "Marital Status",
                        "type": "categorical"
                    },
                    {
                        "name": "Sex",
                        "type": "categorical"                    
                    },
                    {
                        "name": "Hours worked per week",
                        "type": "numerical",
                        "subtype": "integer",
                    },
                    {
                        "name": "Region",
                        "type": "categorical"
                    },
                ],
                "name": "census"
            }
        ]
    }

    tables = {
        'census': df
    }

    sdv = SDV()
    sdv.fit(metadata, tables)
    samples = sdv.sample_all(len(df))
    synth = samples['census']

    assert synth.shape == (569741, 4)
    assert set(synth['Marital Status']) == set(df['Marital Status'])
    assert set(synth['Sex']) == set(df['Sex'])
    assert set(synth['Region']) == set(df['Region'])
