import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils import run

FILESTEM = 'artificial_5-resampling-ensemble'

def input_json(random_state):
    return {
        "enabled": True,
        "dataset": "generator-outputs/artificial/artificial_5",
        "synth-method": "synthpop",
        "parameters": {
            "enabled": True,
            "num_samples_to_fit": -1,
            "num_samples_to_synthesize": -1,
            "num_datasets_to_synthesize": 1,
            "random_state": int(random_state),
            "vars_sequence": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            "synthesis_methods": [
                "sample",
                "sample",
                "sample",
                "sample",
                "sample",
                "sample",
                "sample",
                "sample",
                "sample",
                "sample",
                "sample",
                "sample",
                "sample",
                "sample",
                "sample",
                "sample"
            ],
            "proper": False,
            "tree_minbucket": 1,
        },
        "privacy_parameters_disclosure_risk": {
            "enabled": False,
            "num_samples_intruder": 5000,
            "vars_intruder": ["gender", "age", "neighborhood"],
        },
        "utility_parameters_classifiers": {
            "enabled": False,
            "classifier": {
                "LogisticRegression": {"mode": "main", "params_main": {"max_iter": 1000}}
            },
        },
        "utility_parameters_correlations": {"enabled": False},
        "utility_parameters_feature_importance": {
            "enabled": True,
            "label_column": "Label",
            "normalized_entities": [],
            "max_depth": 2,
            "aggPrimitives": [],
            "tranPrimitives": [],
            "drop_na": True,
            "drop_full_na_columns": False,
            "compute_shapley": True,
            "skip_feature_engineering": True,
            "categorical_enconding": "labels"
        },
    }


if __name__ == "__main__":

    run(input_json, FILESTEM)

