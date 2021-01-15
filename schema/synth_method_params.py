from pydantic import BaseModel


class BaseParameters(BaseModel):

    enabled: bool


class CTGANParameters(BaseModel):

    enabled: bool = True
    num_samples_to_fit: int
    num_samples_to_synthesize: int
    num_datasets_to_synthesize: int
    num_epochs: int
    random_state: int
