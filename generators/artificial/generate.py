"""
Script that generates artificial data using scikit learn for testing the utility of various algorithms

"""

import argparse
import json
import os
import sys

from privgem import tabular_artificial

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
from provenance import generate_provenance_json


def main(output_dir: str):

    # Generate baseline dataset
    n_samples = 10000
    # Classes
    n_classes = 2
    class_weights = [0.5, 0.5]
    n_clusters_per_class = 1
    # Features
    n_features = 5
    n_informative = 5
    n_redundant = 0
    n_repeated = 0
    # Control "noise"
    flip_y = 0.1
    class_sep = 1.0

    # number of categorical columns and their bins
    n_categorical = 5
    n_categorical_bins = [5, 5, 5, 5, 5]

    X, y, categories = \
        tabular_artificial.make_table(n_samples=n_samples,
                                      n_classes=n_classes,
                                      class_weights=class_weights,
                                      n_clusters_per_class=n_clusters_per_class,
                                      n_features=n_features,
                                      n_informative=n_informative,
                                      n_redundant=n_redundant,
                                      n_repeated=n_repeated,
                                      n_categorical=n_categorical,
                                      n_categorical_bins=n_categorical_bins,
                                      flip_y=flip_y,
                                      class_sep=class_sep)

    # combine into one dataframe
    X['Label'] = y

    # write data file
    data_file = os.path.join(output_dir, "artificial_1") + ".csv"
    X.to_csv(data_file, index=False)
    print('dataset written out to: ' + data_file)

    # construct metadata .json file
    meta = {"columns": [], "provenance": []}
    meta["columns"].append({"name": "X0", "type": "Categorical"})
    meta["columns"].append({"name": "X1", "type": "Categorical"})
    meta["columns"].append({"name": "X2", "type": "Categorical"})
    meta["columns"].append({"name": "X3", "type": "Categorical"})
    meta["columns"].append({"name": "X4", "type": "Categorical"})
    meta["columns"].append({"name": "Label", "type": "Categorical"})

    metadata_file = os.path.join(output_dir, "artificial_1") + ".json"
    with open(metadata_file, "w") as mf:
        json.dump(meta, mf, indent=4, sort_keys=True)
    print('metadata file written out to: ' + metadata_file)

    # Generate dataset with redundant features/multicolinearity
    n_informative = 1
    n_redundant = 4

    X, y, categories = \
        tabular_artificial.make_table(n_samples=n_samples,
                                      n_classes=n_classes,
                                      class_weights=class_weights,
                                      n_clusters_per_class=n_clusters_per_class,
                                      n_features=n_features,
                                      n_informative=n_informative,
                                      n_redundant=n_redundant,
                                      n_repeated=n_repeated,
                                      n_categorical=n_categorical,
                                      n_categorical_bins=n_categorical_bins,
                                      flip_y=flip_y,
                                      class_sep=class_sep)

    # combine into one dataframe
    X['Label'] = y

    # write data file
    data_file = os.path.join(output_dir, "artificial_2") + ".csv"
    X.to_csv(data_file, index=False)
    print('dataset written out to: ' + data_file)

    # construct metadata .json file
    meta = {"columns": [], "provenance": []}
    meta["columns"].append({"name": "X0", "type": "Categorical"})
    meta["columns"].append({"name": "X1", "type": "Categorical"})
    meta["columns"].append({"name": "X2", "type": "Categorical"})
    meta["columns"].append({"name": "X3", "type": "Categorical"})
    meta["columns"].append({"name": "X4", "type": "Categorical"})
    meta["columns"].append({"name": "Label", "type": "Categorical"})

    metadata_file = os.path.join(output_dir, "artificial_2") + ".json"
    with open(metadata_file, "w") as mf:
        json.dump(meta, mf, indent=4, sort_keys=True)
    print('metadata file written out to: ' + metadata_file)

    # Generate dataset with high separation between classes
    n_informative = 5
    n_redundant = 0
    flip_y = 0.0
    class_sep = 10.0

    X, y, categories = \
        tabular_artificial.make_table(n_samples=n_samples,
                                      n_classes=n_classes,
                                      class_weights=class_weights,
                                      n_clusters_per_class=n_clusters_per_class,
                                      n_features=n_features,
                                      n_informative=n_informative,
                                      n_redundant=n_redundant,
                                      n_repeated=n_repeated,
                                      n_categorical=n_categorical,
                                      n_categorical_bins=n_categorical_bins,
                                      flip_y=flip_y,
                                      class_sep=class_sep)

    # combine into one dataframe
    X['Label'] = y

    # write data file
    data_file = os.path.join(output_dir, "artificial_3") + ".csv"
    X.to_csv(data_file, index=False)
    print('dataset written out to: ' + data_file)

    # construct metadata .json file
    meta = {"columns": [], "provenance": []}
    meta["columns"].append({"name": "X0", "type": "Categorical"})
    meta["columns"].append({"name": "X1", "type": "Categorical"})
    meta["columns"].append({"name": "X2", "type": "Categorical"})
    meta["columns"].append({"name": "X3", "type": "Categorical"})
    meta["columns"].append({"name": "X4", "type": "Categorical"})
    meta["columns"].append({"name": "Label", "type": "Categorical"})

    metadata_file = os.path.join(output_dir, "artificial_3") + ".json"
    with open(metadata_file, "w") as mf:
        json.dump(meta, mf, indent=4, sort_keys=True)
    print('metadata file written out to: ' + metadata_file)

    # Generate dataset with one informative and four random features
    n_informative = 1
    n_redundant = 0
    flip_y = 0.1
    class_sep = 1.0

    X, y, categories = \
        tabular_artificial.make_table(n_samples=n_samples,
                                      n_classes=n_classes,
                                      class_weights=class_weights,
                                      n_clusters_per_class=n_clusters_per_class,
                                      n_features=n_features,
                                      n_informative=n_informative,
                                      n_redundant=n_redundant,
                                      n_repeated=n_repeated,
                                      n_categorical=n_categorical,
                                      n_categorical_bins=n_categorical_bins,
                                      flip_y=flip_y,
                                      class_sep=class_sep)

    # combine into one dataframe
    X['Label'] = y

    # write data file
    data_file = os.path.join(output_dir, "artificial_4") + ".csv"
    X.to_csv(data_file, index=False)
    print('dataset written out to: ' + data_file)

    # construct metadata .json file
    meta = {"columns": [], "provenance": []}
    meta["columns"].append({"name": "X0", "type": "Categorical"})
    meta["columns"].append({"name": "X1", "type": "Categorical"})
    meta["columns"].append({"name": "X2", "type": "Categorical"})
    meta["columns"].append({"name": "X3", "type": "Categorical"})
    meta["columns"].append({"name": "X4", "type": "Categorical"})
    meta["columns"].append({"name": "Label", "type": "Categorical"})

    metadata_file = os.path.join(output_dir, "artificial_4") + ".json"
    with open(metadata_file, "w") as mf:
        json.dump(meta, mf, indent=4, sort_keys=True)
    print('metadata file written out to: ' + metadata_file)

    # Generate dataset with 15 informative categorical features
    n_features = 15
    n_informative = 15
    n_categorical = 15
    n_categorical_bins = [5] * 15

    X, y, categories = \
        tabular_artificial.make_table(n_samples=n_samples,
                                      n_classes=n_classes,
                                      class_weights=class_weights,
                                      n_clusters_per_class=n_clusters_per_class,
                                      n_features=n_features,
                                      n_informative=n_informative,
                                      n_redundant=n_redundant,
                                      n_repeated=n_repeated,
                                      n_categorical=n_categorical,
                                      n_categorical_bins=n_categorical_bins,
                                      flip_y=flip_y,
                                      class_sep=class_sep)

    # combine into one dataframe
    X['Label'] = y

    # write data file
    data_file = os.path.join(output_dir, "artificial_5") + ".csv"
    X.to_csv(data_file, index=False)
    print('dataset written out to: ' + data_file)

    # construct metadata .json file
    meta = {"columns": [], "provenance": []}
    meta["columns"].append({"name": "X0", "type": "Categorical"})
    meta["columns"].append({"name": "X1", "type": "Categorical"})
    meta["columns"].append({"name": "X2", "type": "Categorical"})
    meta["columns"].append({"name": "X3", "type": "Categorical"})
    meta["columns"].append({"name": "X4", "type": "Categorical"})
    meta["columns"].append({"name": "X5", "type": "Categorical"})
    meta["columns"].append({"name": "X6", "type": "Categorical"})
    meta["columns"].append({"name": "X7", "type": "Categorical"})
    meta["columns"].append({"name": "X8", "type": "Categorical"})
    meta["columns"].append({"name": "X9", "type": "Categorical"})
    meta["columns"].append({"name": "X10", "type": "Categorical"})
    meta["columns"].append({"name": "X11", "type": "Categorical"})
    meta["columns"].append({"name": "X12", "type": "Categorical"})
    meta["columns"].append({"name": "X13", "type": "Categorical"})
    meta["columns"].append({"name": "X14", "type": "Categorical"})
    meta["columns"].append({"name": "Label", "type": "Categorical"})

    metadata_file = os.path.join(output_dir, "artificial_5") + ".json"
    with open(metadata_file, "w") as mf:
        json.dump(meta, mf, indent=4, sort_keys=True)
    print('metadata file written out to: ' + metadata_file)

    # Generate dataset with 15 informative continuous features
    n_features = 15
    n_informative = 15
    n_categorical = 0
    n_categorical_bins = []

    X, y, categories = \
        tabular_artificial.make_table(n_samples=n_samples,
                                      n_classes=n_classes,
                                      class_weights=class_weights,
                                      n_clusters_per_class=n_clusters_per_class,
                                      n_features=n_features,
                                      n_informative=n_informative,
                                      n_redundant=n_redundant,
                                      n_repeated=n_repeated,
                                      n_categorical=n_categorical,
                                      n_categorical_bins=n_categorical_bins,
                                      flip_y=flip_y,
                                      class_sep=class_sep)

    # combine into one dataframe
    X['Label'] = y

    # write data file
    data_file = os.path.join(output_dir, "artificial_6") + ".csv"
    X.to_csv(data_file, index=False)
    print('dataset written out to: ' + data_file)

    # construct metadata .json file
    meta = {"columns": [], "provenance": []}
    meta["columns"].append({"name": "X0", "type": "ContinuousNumerical"})
    meta["columns"].append({"name": "X1", "type": "ContinuousNumerical"})
    meta["columns"].append({"name": "X2", "type": "ContinuousNumerical"})
    meta["columns"].append({"name": "X3", "type": "ContinuousNumerical"})
    meta["columns"].append({"name": "X4", "type": "ContinuousNumerical"})
    meta["columns"].append({"name": "X5", "type": "ContinuousNumerical"})
    meta["columns"].append({"name": "X6", "type": "ContinuousNumerical"})
    meta["columns"].append({"name": "X7", "type": "ContinuousNumerical"})
    meta["columns"].append({"name": "X8", "type": "ContinuousNumerical"})
    meta["columns"].append({"name": "X9", "type": "ContinuousNumerical"})
    meta["columns"].append({"name": "X10", "type": "ContinuousNumerical"})
    meta["columns"].append({"name": "X11", "type": "ContinuousNumerical"})
    meta["columns"].append({"name": "X12", "type": "ContinuousNumerical"})
    meta["columns"].append({"name": "X13", "type": "ContinuousNumerical"})
    meta["columns"].append({"name": "X14", "type": "ContinuousNumerical"})
    meta["columns"].append({"name": "Label", "type": "Categorical"})

    metadata_file = os.path.join(output_dir, "artificial_6") + ".json"
    with open(metadata_file, "w") as mf:
        json.dump(meta, mf, indent=4, sort_keys=True)
    print('metadata file written out to: ' + metadata_file)

    # Generate dataset with 3 categorical features with good separation between
    # their feature importance values - easy to rank feature importances
    n_features = 15
    n_informative = 5
    n_redundant = 5
    n_categorical = 15
    n_categorical_bins = [5] * 15

    X, y, categories = \
        tabular_artificial.make_table(n_samples=n_samples,
                                      n_classes=n_classes,
                                      class_weights=class_weights,
                                      n_clusters_per_class=n_clusters_per_class,
                                      n_features=n_features,
                                      n_informative=n_informative,
                                      n_redundant=n_redundant,
                                      n_repeated=n_repeated,
                                      n_categorical=n_categorical,
                                      n_categorical_bins=n_categorical_bins,
                                      flip_y=flip_y,
                                      class_sep=class_sep)

    # keep only three features
    X = X[["X0", "X9", "X10"]]
    categories = categories[:3]

    # combine into one dataframe
    X['Label'] = y

    # write data file
    data_file = os.path.join(output_dir, "artificial_7") + ".csv"
    X.to_csv(data_file, index=False)
    print('dataset written out to: ' + data_file)

    # construct metadata .json file
    meta = {"columns": [], "provenance": []}
    meta["columns"].append({"name": "X0", "type": "Categorical"})
    meta["columns"].append({"name": "X9", "type": "Categorical"})
    meta["columns"].append({"name": "X10", "type": "Categorical"})
    meta["columns"].append({"name": "Label", "type": "Categorical"})

    metadata_file = os.path.join(output_dir, "artificial_7") + ".json"
    with open(metadata_file, "w") as mf:
        json.dump(meta, mf, indent=4, sort_keys=True)
    print('metadata file written out to: ' + metadata_file)

    print('preparing metadata...')
    parameters = {}
    meta["provenance"] = generate_provenance_json(__file__, parameters)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate artificial data")
    parser.add_argument("--output-dir", type=str, default=os.getcwd(), help="Output directory")

    args = parser.parse_args()

    main(args.output_dir)
