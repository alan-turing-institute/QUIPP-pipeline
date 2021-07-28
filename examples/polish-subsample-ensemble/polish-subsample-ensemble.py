import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils import run, SynthMethod

FILESTEM = 'polish-subsample-ensemble'

def input_json(random_state, sample_frac):
    return {
        "enabled": True,
        "dataset": "datasets/polish_data_2011/polish_data_2011",
        "synth-method": "subsample",
        "parameters": {
            "enabled": True,
            "frac_samples_to_synthesize": sample_frac,
            "random_state": int(random_state),
        },
        "privacy_parameters_disclosure_risk": {
            "enabled": False,
            "num_samples_intruder": 5000,
            "vars_intruder": ["sex", "age"],
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

    run(input_json, FILESTEM, SynthMethod.SUBSAMPLING)
