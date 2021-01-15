from pydantic import BaseModel, Field
from pathlib import Path
from enum import Enum
from typing import List, Union
from pydantic import ValidationError

from .synth_method_params import BaseParameters, CTGANParameters


class SynthMethods(str, Enum):

    CTGAN = "CTGAN"

class PrivacyParametersDisclosureRisk(BaseModel):

    enabled: bool
    num_samples_intruder: int
    vars_intruder: List[str]


class UtilityParametersClassifiers(BaseModel):

    enabled: bool
    input_columns: List[str]
    label_column: str
    test_train_ratio: float
    num_leaked_rows: int


class UtilityParametersCorrelations(BaseModel):

    enabled: bool


class BaseParams(BaseModel):

    enabled: bool
    dataset: Path
    synth_method: SynthMethods = Field(alias="synth-method")
    parameters: BaseParameters
    privacy_parameters_disclosure_risk: PrivacyParametersDisclosureRisk
    utility_parameters_classifiers: UtilityParametersClassifiers
    utility_parameters_correlations: UtilityParametersCorrelations


class CTGAN(BaseParams):

    synth_method: str = Field(SynthMethods.CTGAN.value, alias="synth-method")
    parameters: CTGANParameters


def get_input_validation_schema(synth_method: SynthMethods) -> Union[CTGAN]:
    """Return a Pydantic Model for synth_method

    Args:
        synth_method (SynthMethods): The type of synth-method

    Returns:
        BaseModel: A Pydantic Model for validating input files
    """
    if synth_method == SynthMethods.CTGAN:
        return CTGAN


def validate_input_json(input_json_path: str):
    """Validate input json

    Args:
        input_json_path (str): Path to input json file

    Raises:
        ValueError: Details of validation error
    """

    def validate_file(schema: BaseModel):

        try:
            params = schema.parse_file(input_json_path)
        except ValidationError as ve:
            raise ValueError(
                f"{input_json_path} is not valid \nDetail:\n {ve.json()}"
            ) from None

        return params

    # Check what type of synth method we are using and get schema to validate model
    global_params = validate_file(BaseParams)
    schema = get_input_validation_schema(global_params.synth_method)

    # Validate input json
    params = validate_file(schema)

    return params