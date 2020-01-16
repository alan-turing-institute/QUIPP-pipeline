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

def test_dimension(df):
    assert get_shape(df) == (32561, 15)

def test_read_data():
    df = read_data('adults')
    assert df.shape == (32561, 15)
    assert df['age'].min() > 16
    assert df['age'].max() < 100
    assert len(df['sex'].unique()) == 2
    assert df.columns[2] == 'fnlwgt'

def test_failing_test():
    assert 1 == 1
