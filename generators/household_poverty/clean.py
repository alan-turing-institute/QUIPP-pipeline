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

    # make two copies
    data1 = data.copy()
    data2 = data.copy()

    # create large dataset
    # drop some columns with little predictive contribution to make dataset more manageable
    data1.drop(columns=["planpri", "public", "noelec", "coopele", "v14a", "hacapo", "hacdor", "male",
                       "hogar_mayor", "hogar_total", "tipovivi1", "tipovivi2", "tipovivi3", "tipovivi4",
                       "tipovivi5", "area1", "area2", "SQBage", "SQBhogar_total", "SQBedjefe",
                       "SQBhogar_nin", "SQBovercrowding", "SQBmeaned", "tamhog",
                       "tamviv", "computer", "refrig", "sanitario1", "sanitario2", "sanitario3",
                       "sanitario5", "sanitario6", "abastaguadentro", "abastaguafuera", "abastaguano",
                       "elimbasu1", "elimbasu2", "elimbasu3", "elimbasu4", "elimbasu5", "elimbasu6", "r4h1",
                       "r4h2", "r4h3", "techozinc", "techoentrepiso", "techocane",
                       "techootro", "hogar_adul", "lugar1", "lugar2", "lugar3", "lugar4", "lugar5", "lugar6",
                       "edjefa", "v2a1", "v18q1"], inplace=True)

    # convert one-hot encoding to label encoding
    onehot_cols = [col for col in data1 if col.startswith('pared')]
    data1["pared"] = np.where(data1[onehot_cols])[1]
    data1.drop(columns=onehot_cols, inplace=True)
    onehot_cols = [col for col in data1 if col.startswith('piso')]
    data1["piso"] = np.where(data1[onehot_cols])[1]
    data1.drop(columns=onehot_cols, inplace=True)
    #onehot_cols = [col for col in data1 if col.startswith('techo')]
    #data1.drop(data1[(data1[onehot_cols].T == 0).all()].index, inplace=True)
    #data1["techo"] = np.where(data1[onehot_cols])[1]
    #data1.drop(columns=onehot_cols, inplace=True)
    onehot_cols = [col for col in data1 if col.startswith('energcocinar')]
    data1["energcocinar"] = np.where(data1[onehot_cols])[1]
    data1.drop(columns=onehot_cols, inplace=True)
    onehot_cols = [col for col in data1 if col.startswith('epared')]
    data1["epared"] = np.where(data1[onehot_cols])[1]
    data1.drop(columns=onehot_cols, inplace=True)
    onehot_cols = [col for col in data1 if col.startswith('etecho')]
    data1["etecho"] = np.where(data1[onehot_cols])[1]
    data1.drop(columns=onehot_cols, inplace=True)
    onehot_cols = [col for col in data1 if col.startswith('eviv')]
    data1["eviv"] = np.where(data1[onehot_cols])[1]
    data1.drop(columns=onehot_cols, inplace=True)
    onehot_cols = [col for col in data1 if col.startswith('estadocivil')]
    data1["estadocivil"] = np.where(data1[onehot_cols])[1]
    data1.drop(columns=onehot_cols, inplace=True)
    onehot_cols = [col for col in data1 if col.startswith('parentesco')]
    data1["parentesco"] = np.where(data1[onehot_cols])[1]
    data1.drop(columns=onehot_cols, inplace=True)
    onehot_cols = [col for col in data1 if col.startswith('instlevel')]
    data1.drop(data1[(data1[onehot_cols].T == 0).all()].index, inplace=True)
    data1["instlevel"] = np.where(data1[onehot_cols])[1]
    data1.drop(columns=onehot_cols, inplace=True)
    #onehot_cols = [col for col in data1 if col.startswith('lugar')]
    #data1["lugar"] = np.where(data1[onehot_cols])[1]
    #data1.drop(columns=onehot_cols, inplace=True)

    # put target column at the end of the data frame
    target_col = data1['Target']
    data1.drop(labels=['Target'], axis=1, inplace=True)
    data1.insert(data1.shape[1], 'Target', target_col)

    # construct metadata1 json file
    meta_hp_dataset = {"columns": [], "provenance": []}
    meta_hp_dataset["columns"].append({"name": "Id", "type": "String"})
    meta_hp_dataset["columns"].append({"name": "rooms", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "v18q", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "r4m1", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "r4m2", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "r4m3", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "r4t1", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "r4t2", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "r4t3", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "escolari", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "rez_esc", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "hhsize", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "cielorazo", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "dis", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "female", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "idhogar", "type": "String"})
    meta_hp_dataset["columns"].append({"name": "hogar_nin", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "dependency", "type": "ContinuousNumerical"})
    meta_hp_dataset["columns"].append({"name": "edjefe", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "meaneduc", "type": "ContinuousNumerical"})
    meta_hp_dataset["columns"].append({"name": "bedrooms", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "overcrowding", "type": "ContinuousNumerical"})
    meta_hp_dataset["columns"].append({"name": "television", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "mobilephone", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "qmobilephone", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "age", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "SQBescolari", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "SQBdependency", "type": "ContinuousNumerical"})
    meta_hp_dataset["columns"].append({"name": "agesq", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "pared", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "piso", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "energcocinar", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "epared", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "etecho", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "eviv", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "estadocivil", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "parentesco", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "instlevel", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "Target", "type": "DiscreteNumerical"})

    data_file = os.path.join(output_dir, output_filename + "_large") + ".csv"
    data1.to_csv(data_file, index=False)
    print('dataset written out to: ' + data_file)

    print('preparing metadata...')
    parameters = {}
    meta_hp_dataset["provenance"] = generate_provenance_json(__file__, parameters)

    metadata_file = os.path.join(output_dir, output_filename + "_large") + ".json"
    with open(metadata_file, "w") as mf:
        json.dump(meta_hp_dataset, mf, indent=4, sort_keys=True)
    print('metadata file written out to: ' + metadata_file)

    # create small dataset
    data2.drop(columns=["planpri", "public", "noelec", "coopele", "v14a", "hacapo", "hacdor", "male",
                       "hogar_mayor", "hogar_total", "tipovivi1", "tipovivi2", "tipovivi3", "tipovivi4",
                       "tipovivi5", "area1", "area2", "SQBescolari", "SQBage", "SQBhogar_total", "SQBedjefe",
                       "SQBhogar_nin", "SQBovercrowding", "SQBdependency", "SQBmeaned", "agesq", "tamhog",
                       "tamviv", "computer", "refrig", "sanitario1", "sanitario2", "sanitario3",
                       "sanitario5", "sanitario6", "abastaguadentro", "abastaguafuera", "abastaguano",
                       "elimbasu1", "elimbasu2", "elimbasu3", "elimbasu4", "elimbasu5", "elimbasu6", "r4h1",
                       "r4h2", "r4h3", "mobilephone", "television", "r4m2", "energcocinar1", "energcocinar2",
                       "energcocinar3", "energcocinar4", "r4t2", "techozinc", "techoentrepiso", "techocane",
                       "techootro", "hogar_adul", "lugar1", "lugar2", "lugar3", "lugar4", "lugar5", "lugar6",
                       "edjefa", "hhsize", "v2a1", "v18q1", "rez_esc"], inplace=True)

    # convert one-hot encoding to label encoding
    onehot_cols = [col for col in data2 if col.startswith('pared')]
    data2["pared"] = np.where(data2[onehot_cols])[1]
    data2.drop(columns=onehot_cols, inplace=True)
    onehot_cols = [col for col in data2 if col.startswith('piso')]
    data2["piso"] = np.where(data2[onehot_cols])[1]
    data2.drop(columns=onehot_cols, inplace=True)
    # onehot_cols = [col for col in data2 if col.startswith('techo')]
    # data2.drop(data2[(data2[onehot_cols].T == 0).all()].index, inplace=True)
    # data2["techo"] = np.where(data2[onehot_cols])[1]
    # data2.drop(columns=onehot_cols, inplace=True)
    # onehot_cols = [col for col in data2 if col.startswith('energcocinar')]
    # data2["energcocinar"] = np.where(data2[onehot_cols])[1]
    # data2.drop(columns=onehot_cols, inplace=True)
    onehot_cols = [col for col in data2 if col.startswith('epared')]
    data2["epared"] = np.where(data2[onehot_cols])[1]
    data2.drop(columns=onehot_cols, inplace=True)
    onehot_cols = [col for col in data2 if col.startswith('etecho')]
    data2["etecho"] = np.where(data2[onehot_cols])[1]
    data2.drop(columns=onehot_cols, inplace=True)
    onehot_cols = [col for col in data2 if col.startswith('eviv')]
    data2["eviv"] = np.where(data2[onehot_cols])[1]
    data2.drop(columns=onehot_cols, inplace=True)
    onehot_cols = [col for col in data2 if col.startswith('estadocivil')]
    data2["estadocivil"] = np.where(data2[onehot_cols])[1]
    data2.drop(columns=onehot_cols, inplace=True)
    onehot_cols = [col for col in data2 if col.startswith('parentesco')]
    data2["parentesco"] = np.where(data2[onehot_cols])[1]
    data2.drop(columns=onehot_cols, inplace=True)
    onehot_cols = [col for col in data2 if col.startswith('instlevel')]
    data2.drop(data2[(data2[onehot_cols].T == 0).all()].index, inplace=True)
    data2["instlevel"] = np.where(data2[onehot_cols])[1]
    data2.drop(columns=onehot_cols, inplace=True)
    # onehot_cols = [col for col in data2 if col.startswith('lugar')]
    # data2["lugar"] = np.where(data2[onehot_cols])[1]
    # data2.drop(columns=onehot_cols, inplace=True)

    # put target column at the end of the data frame
    target_col = data2['Target']
    data2.drop(labels=['Target'], axis=1, inplace=True)
    data2.insert(data2.shape[1], 'Target', target_col)

    # construct metadata json file
    meta_hp_dataset = {"columns": [], "provenance": []}
    meta_hp_dataset["columns"].append({"name": "Id", "type": "String"})
    meta_hp_dataset["columns"].append({"name": "rooms", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "v18q", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "r4m1", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "r4m3", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "r4t1", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "r4t3", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "escolari", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "cielorazo", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "dis", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "female", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "idhogar", "type": "String"})
    meta_hp_dataset["columns"].append({"name": "hogar_nin", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "dependency", "type": "ContinuousNumerical"})
    meta_hp_dataset["columns"].append({"name": "edjefe", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "meaneduc", "type": "ContinuousNumerical"})
    meta_hp_dataset["columns"].append({"name": "bedrooms", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "overcrowding", "type": "ContinuousNumerical"})
    meta_hp_dataset["columns"].append({"name": "qmobilephone", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "age", "type": "DiscreteNumerical"})
    meta_hp_dataset["columns"].append({"name": "pared", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "piso", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "epared", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "etecho", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "eviv", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "estadocivil", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "parentesco", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "instlevel", "type": "Categorical"})
    meta_hp_dataset["columns"].append({"name": "Target", "type": "DiscreteNumerical"})

    data_file = os.path.join(output_dir, output_filename) + ".csv"
    data2.to_csv(data_file, index=False)
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
