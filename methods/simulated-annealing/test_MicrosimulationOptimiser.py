import numpy as np
import os
import pandas as pd
from MicrosimulationOptimiser import MicrosimulationOptimiser


def test_MicrosimulationOptimiser():

    age = pd.read_csv(os.path.join(os.getcwd(), "datasets", "SimpleWorld", "age.csv"))
    sex = pd.read_csv(os.path.join(os.getcwd(), "datasets", "SimpleWorld", "sex.csv"))
    ind = pd.read_csv(os.path.join(os.getcwd(), "datasets", "SimpleWorld", "ind-full.csv"))

    age_conditions = [ind["age"] < 50, ind["age"] >= 50]
    sex_conditions = [ind["sex"] == "m", ind["sex"] == "f"]

    # Now create an array that holds the individuals and how their properties correspond to the constraints
    ind_array = np.array([np.select(age_conditions, range(len(age_conditions)), default=None),
                          np.select(sex_conditions, range(len(sex_conditions)), default=None)]).transpose()


    np.random.seed(1)
    region = 0

    opt = MicrosimulationOptimiser(ind_array, age.to_numpy()[region], sex.to_numpy()[region])

    opt.Tmax = 100
    opt.Tmin = 0.1
    population_ids, error = opt.anneal()

    synthetic_population = ind.iloc[population_ids]


    assert age["a0.49"].iloc[region] ==len(synthetic_population[synthetic_population["age"] < 50])

    assert age["a.50+"].iloc[region]== len(synthetic_population[synthetic_population["age"] >= 50])

    assert sex["m"].iloc[region]==len(synthetic_population[synthetic_population["sex"] == "m"])

    assert sex["f"].iloc[region]==len(synthetic_population[synthetic_population["sex"] == "f"])