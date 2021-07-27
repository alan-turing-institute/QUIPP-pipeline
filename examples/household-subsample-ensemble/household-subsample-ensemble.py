import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils import run, SynthMethod

FILESTEM = 'household-subsample-ensemble'

def input_json(random_state, sample_frac):
    return {
        "enabled": True,
        "dataset": "generator-outputs/household_poverty/train_cleaned",
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
            "entity_index": "Id",
            "label_column": "Target",
            "normalized_entities": [
                {"new_entity_id": "household",
                 "index": "idhogar",
                 "additional_variables": ["pared", "piso", "cielorazo",
                                          "epared", "etecho", "eviv",
                                          "rooms", "r4m1",
                                          "r4m3", "r4t1", "r4t3",
                                          "hogar_nin", "bedrooms", "qmobilephone",
                                          "dependency", "edjefe", "meaneduc",
                                          "overcrowding", "Target"]
                 }
            ],
            "max_depth": 2,
            "aggPrimitives": ["min", "max", "count", "mode", "num_unique", "std", "sum"],
            "target_entity": "household",
            "drop_na": True,
            "drop_full_na_columns": True,
            "na_thresh": 0.30,
            "compute_shapley": True,
            "skip_feature_engineering": False,
            "features_to_exclude": ["idhogar"],
            "filter_hh": True
        }
    }

if __name__ == "__main__":

    run(input_json, FILESTEM, SynthMethod.SUBSAMPLING)