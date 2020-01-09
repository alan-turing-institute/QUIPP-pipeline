import pandas as pd

def read_data(dataset_name):
    if dataset_name == 'adults':
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
            'income'
        ]

        return pd.read_csv(url, names=columns)
    else:
        return 0


def get_shape(df):
    return df.shape