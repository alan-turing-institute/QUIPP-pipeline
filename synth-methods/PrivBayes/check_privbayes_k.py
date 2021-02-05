from DataSynthesizer.lib.PrivBayes import calculate_k
import json
import os
import pandas as pd

script_dir = os.path.dirname(__file__)
data_root_dir = os.path.join(script_dir, "../../datasets/")
dataset_definitions = [
    { "name": "Adult", "csv_path": os.path.join(data_root_dir, "adult_dataset/adult.csv") },
    { "name": "Framingham", "csv_path": os.path.join(data_root_dir, "framingham/framingham_cleaned.csv") }
]
thetas = [4.0]
betas = [0.3, 0.5]
epsilons = [0.0001, 0.001, 0.01, 0.1, 0.4, 1.0, 4.0, 10.0]

def add_priv_bayes_data(dataset_definition):
    df = pd.read_csv(dataset_definition["csv_path"])
    rows_cols = df.shape
    dataset_definition["num_rows"] = rows_cols[0]
    dataset_definition["num_features"] = rows_cols[1]
    dataset_definition["priv_bayes"] = []
    for theta in thetas:
        for beta in betas:
            for epsilon in epsilons:
                k = calculate_k(
                    num_attributes = dataset_definition["num_features"], 
                    num_tuples = dataset_definition["num_rows"],
                    target_usefulness = theta,
                    epsilon = beta*epsilon)
                dataset_definition["priv_bayes"].append({"theta": theta, "beta": beta, "epsilon": epsilon, "k": k})
    return dataset_definition

for dataset_definition in dataset_definitions:
    dataset_definition = add_priv_bayes_data(dataset_definition)
    print("{name} data set ({num_features} features, {num_rows} rows).".format(
        name=dataset_definition["name"],
        num_features=dataset_definition["num_features"],
        num_rows=dataset_definition["num_rows"]))
    df = pd.DataFrame.from_dict(dataset_definition["priv_bayes"])
    pivot = pd.pivot_table(df, values=["k"], index=["theta", "epsilon"], columns=["beta"])
    print(pivot)
