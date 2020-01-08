import numpy as np
import os
import pandas as pd
import pytest
from MicrosimulationOptimiser import MicrosimulationOptimiser


@pytest.fixture
def simpleworld_example():
    """Load the data from the SimpleWorld example"""

    ind = pd.read_csv(os.path.join(os.getcwd(), "datasets", "SimpleWorld", "ind-full.csv"))
    age = pd.read_csv(os.path.join(os.getcwd(), "datasets", "SimpleWorld", "age.csv"))
    sex = pd.read_csv(os.path.join(os.getcwd(), "datasets", "SimpleWorld", "sex.csv"))

    age_conditions = [ind["age"] < 50, ind["age"] >= 50]
    sex_conditions = [ind["sex"] == "m", ind["sex"] == "f"]

    return ind, age, sex, age_conditions, sex_conditions


def test_MicrosimulationOptimiser(simpleworld_example):
    """Run the simulated annealing algorithm on Region 0 in the SimpleWorld example and check that we get the correct
    number of people in each of the categories."""

    ind, age, sex, age_conditions, sex_conditions = simpleworld_example

    region = 0
    np.random.seed(1)

    # Now create an array that holds the individuals and how their properties correspond to the constraints
    ind_array = np.array([np.select(age_conditions, range(len(age_conditions)), default=None),
                          np.select(sex_conditions, range(len(sex_conditions)), default=None)]).transpose()

    opt = MicrosimulationOptimiser(ind_array, age.to_numpy()[region], sex.to_numpy()[region])

    opt.Tmax = 100
    opt.Tmin = 0.1
    population_ids, error = opt.anneal()

    synthetic_population = ind.iloc[population_ids]

    assert age["a0.49"].iloc[region] == len(synthetic_population[synthetic_population["age"] < 50])
    assert age["a.50+"].iloc[region] == len(synthetic_population[synthetic_population["age"] >= 50])
    assert sex["m"].iloc[region] == len(synthetic_population[synthetic_population["sex"] == "m"])
    assert sex["f"].iloc[region] == len(synthetic_population[synthetic_population["sex"] == "f"])