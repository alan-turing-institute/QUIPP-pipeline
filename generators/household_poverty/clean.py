"""
Script that cleans the Household Poverty raw dataset.

Code taken from:
https://www.kaggle.com/willkoehrsen/featuretools-for-good

"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
from provenance import generate_provenance_json


def main(output_dir: str, output_filename: str):

    # We expect all the data to be in a "data" folder at the same directory level as this script.
    # The postcodes file is the only exception. As it is so large,
    # we sometimes supply an alternative (with full path).
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "train.csv")

    # Raw data
    data = pd.read_csv(data_dir)

    # ### Data Preprocessing

    # Groupby the household and figure out the number of unique values
    all_equal = data.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)

    # Households where targets are not all equal
    not_equal = all_equal[all_equal != True]
    print('There are {} households where the family members do not all have the same target.'.format(len(not_equal)))

    # Iterate through each household
    for household in not_equal.index:
        # Find the correct label (for the head of household)
        true_target = int(data[(data['idhogar'] == household) & (data['parentesco1'] == 1.0)]['Target'])

        # Set the correct label for all members in the household
        data.loc[data['idhogar'] == household, 'Target'] = true_target

    # Groupby the household and figure out the number of unique values
    all_equal = data.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)

    # Households where targets are not all equal
    not_equal = all_equal[all_equal != True]
    print('There are {} households where the family members do not all have the same target.'.format(len(not_equal)))

    # Here we extract the train labels for the heads of households.
    train_valid = data.loc[data['parentesco1'] == 1, ['idhogar', 'Target']].copy()

    # ### Convert object types to floats
    # The mapping is explained in the data description. These are
    # continuous variables and should be converted to numeric floats.
    mapping = {"yes": 1, "no": 0}

    # Fill in the values with the correct mapping
    data['dependency'] = data['dependency'].replace(mapping).astype(np.float64)
    data['edjefa'] = data['edjefa'].replace(mapping).astype(np.float64)
    data['edjefe'] = data['edjefe'].replace(mapping).astype(np.float64)

    # ### Handle Missing Values
    # The logic for these choices is explained in the
    # [Start Here: A Complete Walkthrough](https://www.kaggle.com/willkoehrsen/start-here-a-complete-walkthrough)
    # kernel. This might not be optimal, but it has improved cross-validation results.
    data['v18q1'] = data['v18q1'].fillna(0)
    data.loc[(data['tipovivi1'] == 1), 'v2a1'] = 0
    data['rez_esc'] = data['rez_esc'].fillna(0)

    # construct metadate json file
    meta_hp_dataset = {"columns": [], "provenance": []}

    meta_hp_dataset["columns"].append({"name": "Id", "type": "String"})
    meta_hp_dataset["columns"].append({"name": "v2a1", "type": "ContinuousNumerical"})
    meta_hp_dataset["columns"].append({"name": "hacdor", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "rooms", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "hacapo", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "refrig", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "v18q", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "v18q1", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "r4h1", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "r4h2", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "r4h3", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "r4m1", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "r4m2", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "r4m3", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "r4t1", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "r4t2", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "r4t3", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "tamhog", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "tamviv", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "escolari", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "rez_esc", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "hhsize", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "paredblolad", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "paredzocalo", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "paredpreb", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "pareddes", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "paredmad", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "paredzinc", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "paredfibras", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "paredother", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "pisomoscer", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "pisocemento", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "pisoother", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "pisonatur", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "pisonotiene", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "pisomadera", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "techozinc", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "techoentrepiso", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "techocane", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "techootro", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "cielorazo", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "abastaguadentro", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "abastaguafuera", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "abastaguano", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "public", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "planpri", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "noelec", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "coopele", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "sanitario1", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "sanitario2", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "sanitario3", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "sanitario5", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "sanitario6", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "energcocinar1", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "energcocinar2", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "energcocinar3", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "energcocinar4", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "elimbasu1", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "elimbasu2", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "elimbasu3", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "elimbasu4", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "elimbasu5", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "elimbasu6", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "epared1", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "epared2", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "epared3", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "eviv1", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "eviv2", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "eviv3", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "dis", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "male", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "female", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "estadocivil1", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "estadocivil2", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "estadocivil3", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "estadocivil4", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "estadocivil5", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "estadocivil6", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "estadocivil7", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "parentesco1", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "parentesco2", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "parentesco3", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "parentesco4", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "parentesco5", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "parentesco6", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "parentesco7", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "parentesco8", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "parentesco9", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "parentesco10", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "parentesco11", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "parentesco12", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "idhogar", "type": "String"})
    meta_hp_dataset["columns"].append({"name": "hogar_nin", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "hogar_adul", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "hogar_mayor", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "hogar_total", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "dependency", "type": "ContinuousNumerical"})
    meta_hp_dataset["columns"].append({"name": "edjefe", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "edjefa", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "meaneduc", "type": "ContinuousNumerical"})
    meta_hp_dataset["columns"].append({"name": "instlevel1", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "instlevel2", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "instlevel3", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "instlevel4", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "instlevel5", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "instlevel6", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "instlevel7", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "instlevel8", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "instlevel9", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "bedrooms", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "overcrowding", "type": "ContinuousNumerical"})
    meta_hp_dataset["columns"].append({"name": "tipovivi1", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "tipovivi2", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "tipovivi3", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "tipovivi4", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "tipovivi5", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "computer", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "television", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "mobilephone", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "qmobilephone", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "lugar1", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "lugar2", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "lugar3", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "lugar4", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "lugar5", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "lugar6", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "area1", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "area2", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "age", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "SQBescolari", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "SQBage", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "SQBhogar_total", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "SQBedjefe", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "SQBhogar_nin", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "SQBovercrowding", "type": "ContinuousNumerical"})
    meta_hp_dataset["columns"].append({"name": "SQBdependency", "type": "ContinuousNumerical"})
    meta_hp_dataset["columns"].append({"name": "SQBmeaned", "type": "ContinuousNumerical"})
    meta_hp_dataset["columns"].append({"name": "agesq", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "Target", "type": "DiscreteNumerical"})

    data_file = os.path.join(output_dir, output_filename) + ".csv"
    data.to_csv(data_file, index=False)
    print('dataset written out to: ' + data_file)

    print('preparing metadata...')
    parameters = {}
    meta_hp_dataset["provenance"] = generate_provenance_json(__file__, parameters)

    metadata_file = os.path.join(output_dir, output_filename) + ".json"
    with open(metadata_file, "w") as mf:
        json.dump(meta_hp_dataset, mf, indent=4, sort_keys=True)
    print('metadata file written out to: ' + metadata_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Clean household poverty data")
    parser.add_argument("--output-dir", type=str, default=os.getcwd(), help="Output directory")
    parser.add_argument("--output-filename", type=str, default='train_cleaned',
                        help="Output filename (without extension")
    args = parser.parse_args()

    main(args.output_dir, args.output_filename)
