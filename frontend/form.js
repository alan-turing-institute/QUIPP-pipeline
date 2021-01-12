const Form = JSONSchemaForm.default;

const schema =
{
    "$schema": "http://json-schema.org/draft-07/schema",
    "type": "object",
    "title": "The parameter input file",
    "description": "Parameters passed to the synthesis method, and the utility and privacy metrics",
    "required": [
        "dataset",
        "synth-method",
        "num_samples_to_fit",
        "num_samples_to_synthesize",
        "num_datasets_to_synthesize",
        "random_state",
        "disclosure_risk",
        "sklearn_classifiers"
    ],
    "dependencies": {
        "synth-method": {
            "oneOf": [
                {
                    "required": ["sgf"],
                    "properties": {
                        "synth-method": { "enum": ["sgf"] },
                        "sgf": {
                            "type": "object",
                            "title": "SGF parameters",
                            "description": "Parameters for the SGF synthesis method",
                            "required": [
                                "gamma",
                                "omega",
                                "ncomp",
                                "ndist",
                                "k",
                                "epsilon0",
                                "tinc"
                            ],
                            "properties": {
                                "gamma": {
                                    "type": "number",
                                    "title": "gamma",
                                    "description": "Privacy parameter which controls how the plausible seeds for a record are determined.",
                                    "exclusiveMinimum": 1.0
                                },
                                "omega": {
                                    "type": "integer",
                                    "title": "omega",
                                    "description": "The number of resampled attributes is N - omega",
                                    "minimum": 0
                                },
                                "ncomp": {
                                    "enum": ["adv", "seq"],
                                    "title": "Composition strategy",
                                    "description": "",
                                    "default": "adv"
                                },
                                "ndist": {
                                    "enum": ["none", "lap", "geom"],
                                    "title": "Noise distribution",
                                    "description": "The noise distribution of the generative model: 'none', 'lap' (Laplacian) or 'geom' (Geometric)."
                                },
                                "k": {
                                    "type": "integer",
                                    "title": "k",
                                    "description": "Minimum number of plausible seeds that a synthetic data point needs to have to be released.",
                                    "minimum": 1
                                },
                                "epsilon0": {
                                    "type": "number",
                                    "title": "epsilon0",
                                    "description": "ε-Differential privacy parameter"
                                },
                                "tinc": {
                                    "type": "integer",
                                    "title": "tinc",
                                    "description": "Step size to create trade-off curve (between 1 and k - 1)"
                                }
                            }
                        }
                    }
                },
                {
                    "properties": {
                        "synth-method": { "enum": ["synthpop"] }
                    }
                }
            ]
        }
    },
    "properties": {
        "dataset": {
            "type": "string",
            "title": "The dataset",
            "description": "The prefix of the filename of the dataset (.csv will be appended)"
        },
        "synth-method": {
            "enum": ["ctgan", "sgf", "synthpop"],
            "title": "Synthesis method",
            "description": "The synthesis method used for the run.  It must correspond to a subdirectory of `synth-methods`"
        },
        "num_samples_to_fit": {
            "type": "integer",
            "title": "Number of samples to fit",
            "description": "How many samples from the input dataset should be used as input to the synthesis procedure?  To use all of the input records, pass a value of `-1`"
        },
        "num_samples_to_synthesize": {
            "type": "integer",
            "title": "Number of samples to synthesize",
            "description": "How many synthetic samples should be produced as output?  To produce the same number of output records as input records, pass a value of `-1`."
        },
        "num_datasets_to_synthesize": {
            "type": "integer",
            "title": "Number of entire datasets to synthesize",
            "description": "How many entire synthetic datasets should be produced?",
            "minimum": 0
        },
        "random_state": {
            "type": "integer",
            "title": "Random seed",
            "description": "the seed for the random number generator (most methods require a PRNG: the seed can be explicitly passed to aid with the testability and reproducibility of the synthetic output)",
            "default": 0
        },
        "disclosure_risk": {
            "type": "object",
            "title": "Disclosure risk parameters",
            "description": "Parameters passed to the disclosure risk privacy metric",
            "required": [
                "num_samples_intruder",
                "vars_intruder"
            ],
            "properties": {
                "num_samples_intruder": {
                    "type": "integer",
                    "title": "Intruder sample count",
                    "description": "How many records corresponding to the original dataset exist in a dataset visible to an attacker?",
                    "minimum": 0
                },
                "vars_intruder": {
                    "type": "array",
                    "title": "Intruder variables",
                    "description": "Names of the columns that are available in the attacker-visible dataset",
                    "items": { "type": "string" }
                }
            }
        },
        "sklearn_classifiers": {
            "type": "object",
            "title": "The Sklearn_classifiers Schema",
            "description": "Parameters needed to compute the classification utility scores with scikit learn",
            "required": [
                "input_columns",
                "label_column",
                "test_train_ratio",
                "num_leaked_rows"
            ],
            "properties": {
                "input_columns": {
                    "type": "array",
                    "items": { "type": "string" },
                    "title": "Column names",
                    "description": "Names of the columns to use as the explanatory variables for the classification"
                },
                "label_column": {
                    "type": "string",
                    "title": "Label column",
                    "description": "The name of the column to use for the category labels"
                },
                "test_train_ratio": {
                    "type": "number",
                    "title": "test/train ratio",
                    "description": "Fraction of records to use in the test set for the classification",
                    "minimum": 0.0
                },
                "num_leaked_rows": {
                    "type": "integer",
                    "title": "Count of leaked records",
                    "description": "The number of additional records from the original dataset with which to augment the synthetic data set before training the classifiers. This is primarily an option to enable testing of the utility metric (i.e. the more rows we leak, the better the utility should become). It should be set to 0 during normal synthesis tasks.",
                    "default": 0,
                    "minimum": 0
                }
            }
        }
    }
}
    
// const log = (type) => console.log.bind(console, type);

function download(filename, text) {
  var element = document.createElement('a');
  element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text));
  element.setAttribute('download', filename);

  element.style.display = 'none';
  document.body.appendChild(element);

  element.click();

  document.body.removeChild(element);
}

ReactDOM.render(
    React.createElement(Form, {
        schema: schema,
        onSubmit: (input) =>
            download("example.json", JSON.stringify(input.formData, null, 4)),
    }),
    document.getElementById("app"));
