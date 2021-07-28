import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils import run, SynthMethod

FILESTEM = 'adult-subsample-ensemble'

def input_json(random_state, sample_frac):
    return {
        "enabled": True,
        "dataset": "datasets/adult_dataset/adult",
        "synth-method": "subsample",
        "parameters": {
            "enabled": True,
            "frac_samples_to_synthesize": sample_frac,
            "random_state": int(random_state),
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
            "label_column": "label",
            "normalized_entities": [
                {
                    "new_entity_id": "education",
                    "index": "education-num",
                    "additional_variables": ["education"],
                    "make_time_index": False,
                },
                {
                    "new_entity_id": "Workclass",
                    "index": "workclass",
                    "additional_variables": [],
                    "make_time_index": False,
                },
                {
                    "new_entity_id": "Occupation",
                    "index": "occupation",
                    "additional_variables": [],
                    "make_time_index": False,
                },
            ],
            "max_depth": 2,
            "features_to_exclude": ["education-num"],
            "drop_na": True,
            "categorical_enconding": "labels",
            "compute_shapley": True,
            "skip_feature_engineering": False
        },
    }


if __name__ == "__main__":

    run(input_json, FILESTEM, SynthMethod.SUBSAMPLING)