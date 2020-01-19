# # SDV pipeline
# 
# SDV (Synthetic Data Vault) is a Python library (https://github.com/HDI-Project/SDV) that allows users to statistically model an entire multi-table, relational dataset. Users can then use the statistical model to generate a synthetic dataset. 
# 
# Underneath the hood it uses a unique hierarchical generative modeling method and recursive sampling techniques. Specifically, it fits a parametric models (Normal, uniform, etc) to each variable and uses a multivariate version of the Gaussian Copula to capture covariances between variables.
# 
# The library is designed to handle complete relational databases by sharing information between tables that have common keys.
# 
# The library has a function to automatically recognise the data types of each variable and choose a model accordingly but it does not work very well so it is safer if the user provides the necessary metadata in the form of a distionary.
# 
# In their paper (https://dai.lids.mit.edu/wp-content/uploads/2018/03/SDV.pdf), they describe an interesting experiment where they hired  DSs to analyse the original and synthetic data set to see how the results differ (which is a form of utility quantification).
# 
# There is one privacy feature that distorts values that the user characterises as private.
# 
# A first tes shows that the synthetic data might not always look like to original in terms of their distribution.

# ### Import libraries
import pandas as pd
import util
from sdv import SDV
from sdv.evaluation import evaluate


# ### Read the data
# Read a census data set from a website
df = util.read_data('adults')
df.head()


# ### Create metadata
# This is a list of the tables in the database 
tables = {
    'census': df
}

# This is a list of their variables and their data types in the form of dictionary - see docs for more details
metadata = {
    "tables": [
        {
            "fields": [
                {
                    "name": "age",
                    "type": "numerical",
                    "subtype": "integer",
                },
                {
                    "name": "workclass",
                    "type": "categorical",
                },
                {
                    "name": "fnlwgt",
                    "type": "numerical",
                    "subtype": "integer",
                },
                {
                    "name": "education",
                    "type": "categorical",
                },
                {
                    "name": "education-num",
                    "type": "numerical",
                    "subtype": "integer",
                },
                {
                    "name": "marital-status",
                    "type": "categorical",
                },
                {
                    "name": "occupation",
                    "type": "categorical",
                },
                {
                    "name": "relationship",
                    "type": "categorical",
                },
                {
                    "name": "race",
                    "type": "categorical",
                },
                {
                    "name": "sex",
                    "type": "categorical",
                },
                {
                    "name": "capital-gain",
                    "type": "numerical",
                    "subtype": "integer",
                },
                {
                    "name": "capital-loss",
                    "type": "numerical",
                    "subtype": "integer",
                },
                {
                    "name": "hours-per-week",
                    "type": "numerical",
                    "subtype": "integer",
                },
                {
                    "name": "native-country",
                    "type": "categorical",
                },
                {
                    "name": "income",
                    "type": "categorical",
                }
            ],
            "name": "census",
        }
    ]
}


# ### Fit the model
# Create an SDV object and fit the models to the data 
sdv = SDV()
sdv.fit(metadata, tables)


# ### Sample synthetic data
# Create a synthetic data set with equal number of rows as the original
samples = sdv.sample('census', num_rows=len(df))
samples['census'].head()

df.shape, samples['census'].shape


# ### Utility metrics
# Use the evaluate function of the package to compare the original and synthetic data sets
# Unclear what it does exactly from the documentation - have to look at the code
evaluate(samples, real=tables, metadata=sdv.metadata)

df.describe()

samples['census'].describe()

# Histograms show that there are some discrepancies in frequencies - under investigation
df['relationship'].hist()
samples['census']['relationship'].hist()


# ### Synthesis on the ONS census data
df2 = pd.read_csv("../../datasets/rft-teaching-file/2011 Census Microdata Teaching File.csv", header=1)
df2.head()

metadata2 = {
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


tables2 = {
    'census': df2
}

sdv2 = SDV()
sdv2.fit(metadata2, tables2)

samples2 = sdv2.sample_all(len(df2))
synth = samples2['census']

# Again we see discrepancies in the frequencies
df2['Marital Status'].hist()

synth['Marital Status'].hist()


