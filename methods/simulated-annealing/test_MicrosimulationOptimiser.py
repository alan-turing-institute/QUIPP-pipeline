import numpy as np
import os
import pandas as pd
import pytest
from MicrosimulationOptimiser import MicrosimulationOptimiser


@pytest.fixture
def simpleworld_example():
    """Load the data from the SimpleWorld example"""

    this_directory = os.path.dirname(os.path.abspath(__file__))

    ind = pd.read_csv(os.path.join(this_directory, "..", "..", "datasets", "SimpleWorld", "ind-full.csv"))
    age = pd.read_csv(os.path.join(this_directory, "..", "..", "datasets", "SimpleWorld", "age.csv"))
    sex = pd.read_csv(os.path.join(this_directory, "..", "..", "datasets", "SimpleWorld", "sex.csv"))

    age_conditions = [ind["age"] < 50, ind["age"] >= 50]
    sex_conditions = [ind["sex"] == "m", ind["sex"] == "f"]

    return ind, age, sex, age_conditions, sex_conditions


# Hardcode region and counts of <50 years, >=50 years, male and female.
# In this simple problem, the optimiser should be able to exactly match these hardcoded expected values.
region_constraints = [(0, 8, 4, 6, 6),
                      (1, 2, 8, 4, 6),
                      (2, 7, 4, 3, 8)]


@pytest.mark.parametrize("region,young,old,male,female", region_constraints)
def test_MicrosimulationOptimiser(region, young, old, male, female, simpleworld_example):
    """Run the simulated annealing algorithm on Region 0 in the SimpleWorld example and check that we get the correct
    number of people in each of the categories."""

    np.random.seed(1)
    ind, age, sex, age_conditions, sex_conditions = simpleworld_example

    # Give properties of below/above 50 and male/female to each individual to match constraints
    ind_array = np.array([np.select(age_conditions, range(len(age_conditions)), default=None),
                          np.select(sex_conditions, range(len(sex_conditions)), default=None)]).transpose()

    # Initialise the optimisation class with the array of individuals and arrays of age and sex constraints (counts)
    opt = MicrosimulationOptimiser(ind_array, age.to_numpy()[region], sex.to_numpy()[region])

    # We'll use the same temperature settings as in the example shown in the notebook
    opt.Tmax = 100
    opt.Tmin = 0.1

    # Annealing process swaps individuals until constraints are met, then we extract the IDs of the synthetic population
    population_ids, error = opt.anneal()
    synthetic_population = ind.iloc[population_ids]

    # Compare numbers of people with each characteristic in the synthetic population with the expected values
    assert len(synthetic_population[synthetic_population["age"] < 50]) == young
    assert len(synthetic_population[synthetic_population["age"] >= 50]) == old
    assert len(synthetic_population[synthetic_population["sex"] == "m"]) == male
    assert len(synthetic_population[synthetic_population["sex"] == "f"]) == female
