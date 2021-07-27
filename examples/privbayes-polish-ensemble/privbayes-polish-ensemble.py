import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils import run, SynthMethod


FILESTEM = 'privbayes-polish-ensemble'

def input_json(random_state, epsilon, k):
    return {
        "enabled": True,
        "dataset": "datasets/polish_data_2011/polish_data_2011",
        "synth-method": "PrivBayes",
        "parameters": {
            "enabled": True,
            "num_samples_to_synthesize": 4897,
            "random_state": int(random_state),
            "category_threshold": 20,
            "epsilon": epsilon,
            "k": int(k),
            "keys": [],
            "histogram_bins": 10,
            "preconfigured_bn": {},
            "save_description": False
        },
        "privacy_parameters_disclosure_risk": {"enabled": False},
        "utility_parameters_classifiers": {
            "enabled": False,
            "classifier": {
                "LogisticRegression": {"mode": "main", "params_main": {"max_iter": 1000}}
            }
        },
        "utility_parameters_correlations": {"enabled": False},
        "utility_parameters_feature_importance": {
            "enabled": True,
            "label_column": "wkabint",
            "normalized_entities": [],
            "max_depth": 2,
            "aggPrimitives": [],
            "tranPrimitives": ["multiply_numeric", "subtract_numeric",
                               "add_numeric", "divide_numeric",
                               "percentile"],
            "drop_na": True,
            "drop_full_na_columns": False,
            "compute_shapley": True,
            "skip_feature_engineering": False,
            "categorical_enconding": "labels"
        }
    }

if __name__ == "__main__":

    run(input_json, FILESTEM, SynthMethod.PRIVBAYES)
