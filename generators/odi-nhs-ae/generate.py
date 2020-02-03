"""
Script that generates hospital A&E data to use in the synthetic data tutorial.

Columns of data inspired by NHS+ODI Leeds blog post:
https://odileeds.org/blog/2019-01-24-exploring-methods-for-creating-synthetic-a-e-data

"""

import argparse
import os
from datetime import datetime
import random
import string
import time

import pandas as pd
import numpy as np

import filepaths


def main(num_rows, filename, seed, postcode_file="London postcodes.csv"):
    print('generating data...')
    start = time.time()

    np.random.seed(seed)
    random.seed(seed)

    hospital_ae_dataset = {}

    print('generating Health Service ID numbers...')
    hospital_ae_dataset['Health Service ID'] = generate_health_service_id_numbers(num_rows)

    print('generating patient ages and times in A&E...')
    (hospital_ae_dataset['Age'], hospital_ae_dataset['Time in A&E (mins)']) = generate_ages_times_in_age(num_rows)

    print('generating hospital instances...')
    hospital_ae_dataset['Hospital'] = generate_hospitals(num_rows)

    print('generating arrival times...')
    hospital_ae_dataset['Arrival Time'] = generate_arrival_times(num_rows)

    print('generating A&E treaments...')
    hospital_ae_dataset['Treatment'] = generate_treatments(num_rows)

    print('generating patient gender instances...')
    hospital_ae_dataset['Gender'] = generate_genders(num_rows)

    print('generating patient postcodes...')
    hospital_ae_dataset['Postcode'] = generate_postcodes(num_rows, os.path.join(filepaths.data_dir, postcode_file))

    filepath = filepaths.output_dir + '/' + filename
    write_out_dataset(hospital_ae_dataset, filepath)
    print('dataset written out to: ' + filepath)

    elapsed = round(time.time() - start, 2)
    print('done in ' + str(elapsed) + ' seconds.')


def generate_ages_times_in_age(num_of_rows) -> (list, list):
    """
    Generates correlated ages and waiting times and returns them as lists

    Obviously normally distributed ages is not very true to real life but is fine for our mock data.

    Correlated random data generation code based on:
    https://realpython.com/python-random/
    """
    # Start with a correlation matrix and standard deviations.
    # 0.9 is the correlation between ages and waiting times, and the correlation of a variable with itself is 1
    correlations = np.array([[1, 0.95], [0.95, 1]])

    # Standard deviations/means of ages and waiting times, respectively
    stdev = np.array([20, 20])
    mean = np.array([41, 60])
    cov = corr2cov(correlations, stdev)

    data = np.random.multivariate_normal(mean=mean, cov=cov, size=num_of_rows)
    data = np.array(data, dtype=int)

    # negative ages or waiting times wouldn't make sense
    # so set any negative values to 0 and 1 respectively 
    data[np.nonzero(data[:, 0] < 1)[0], 0] = 0
    data[np.nonzero(data[:, 1] < 1)[0], 1] = 1

    ages = data[:, 0].tolist()
    times_in_ae = data[:, 1].tolist()

    return ages, times_in_ae


def corr2cov(correlations: np.ndarray, stdev: np.ndarray) -> np.ndarray:
    """Covariance matrix from correlation & standard deviations"""
    diagonal_stdev = np.diag(stdev)
    covariance = diagonal_stdev @ correlations @ diagonal_stdev
    return covariance


def generate_admission_ids(num_of_rows) -> list:
    """ Generate a unique 10-digit ID for every admission record """
    
    uids = []
    for _ in range(num_of_rows):    
        x = ''.join(random.choice(string.digits) for _ in range(10))
        uids.append(x)
    return uids


def generate_health_service_id_numbers(num_of_rows) -> list:
    """ Generate dummy Health Service ID numbers similar to NHS 10 digit format
    See: https://www.nhs.uk/using-the-nhs/about-the-nhs/what-is-an-nhs-number/
    """
    health_service_id_numbers = []
    for _ in range(num_of_rows): 
        health_service_id = ''.join(random.choice(string.digits) for _ in range(3)) + '-'   
        health_service_id += ''.join(random.choice(string.digits) for _ in range(3)) + '-'   
        health_service_id += ''.join(random.choice(string.digits) for _ in range(4))
        health_service_id_numbers.append(health_service_id)
    return health_service_id_numbers


def generate_postcodes(num_of_rows, postcode_file_path) -> list:
    """ Reads a .csv containing info on every London postcode. Reads the 
    postcodes in use and returns a sample of them.

    # List of London postcodes from https://www.doogal.co.uk/PostcodeDownloads.php
    """
    postcodes_df = pd.read_csv(postcode_file_path)
    postcodes_in_use = list(postcodes_df[postcodes_df['In Use?'] == "No"]['Postcode'])
    postcodes = random.choices(postcodes_in_use, k=num_of_rows)
    return postcodes


def generate_hospitals(num_of_rows) -> list:
    """ Reads the data/hospitals_london.txt file, and generates a
    sample of them to add to the dataset.

    List of London hospitals loosely based on 
    https://en.wikipedia.org/wiki/Category:NHS_hospitals_in_London
    """
    with open(filepaths.hospitals_london, 'r') as file_in:
        hospitals = file_in.readlines()
    hospitals = [name.strip() for name in hospitals]

    weights = random.choices(range(1, 100), k=len(hospitals))
    hospitals = random.choices(hospitals, k=num_of_rows, weights=weights)

    return hospitals


def generate_arrival_times(num_of_rows) -> list:
    """ Generate and return arrival times.
        Hardcoding times to first week of April 2019
    """
    arrival_times = []

    # first 7 days in April 2019
    days_dates = [1, 2, 3, 4, 5, 6, 7]
    # have more people come in at the weekend - higher weights 
    day_weights = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1]
    days = random.choices(days_dates, day_weights, k=num_of_rows)
    # this is just so each day has a different peak time
    days_time_modes = {day: random.random() for day in days_dates}

    for day in days:
        start = datetime(2019, 4, day, 00, 00, 00)
        end = datetime(2019, 4, day, 23, 59, 59)

        random_num = random.triangular(0, 1, days_time_modes[day])
        random_datetime = start + (end - start) * random_num
        arrival_times.append(random_datetime.strftime('%Y-%m-%d %H:%M:%S'))

    return arrival_times


def generate_genders(num_of_rows) -> list:
    """ Generate and return list of genders for every row. 

    # National codes for gender in NHS data
    # https://www.datadictionary.nhs.uk/data_dictionary/attributes/p/person/person_gender_code_de.asp?shownav=1
    """
    gender_codes_df = pd.read_csv(filepaths.nhs_ae_gender_codes)
    genders = gender_codes_df['Gender'].tolist()
    # these weights are just dummy values. please don't take them as accurate.
    weights =[0.005, 0.495, 0.495, 0.005]
    gender_codes = random.choices(genders, k=num_of_rows, weights=weights)
    return gender_codes


def generate_treatments(num_of_rows) -> list:
    """ Generate and return sample of treatments patients received. 

    Reads data/treatment_codes_nhs_ae.csv file 

    NHS treatment codes:
    https://www.datadictionary.nhs.uk/web_site_content/supporting_information/clinical_coding/accident_and_emergency_treatment_tables.asp?shownav=1
    """

    treatment_codes_df = pd.read_csv(filepaths.nhs_ae_treatment_codes)
    treatments = treatment_codes_df['Treatment'].tolist()

    # likelihood of each of the treatments - make some more common
    weights = random.choices(range(1, 100), k=len(treatments))
    treatment_codes = random.choices(
        treatments, k=num_of_rows, weights=weights)
    return treatment_codes


def write_out_dataset(dataset: dict, filepath: str):
    """Writing dataset to .csv file

    Keyword arguments:
    dataset -- the dataset to be written to disk
    filepath -- path to write the file out to
    """

    df = pd.DataFrame.from_dict(dataset)
    df.to_csv(filepath, index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate synthetic NHS A&E admissions data")
    parser.add_argument("--rows", type=int, default=10000, help="Number of rows to generate")
    parser.add_argument("--filename", type=str, default='hospital_ae_data.csv', help="Output data filename")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    args = parser.parse_args()

    main(args.rows, args.filename, args.seed)
