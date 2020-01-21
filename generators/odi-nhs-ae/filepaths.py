import os
from pathlib import Path

this_filepath = Path(os.path.realpath(__file__))
project_root = str(this_filepath.parents[0])

data_dir = os.path.join(project_root, "data")
output_dir = os.path.join(project_root, "..", "..", "datasets", "generated", "odi_nhs_ae")

postcodes_london = os.path.join(data_dir, 'London postcodes.csv')
hospitals_london = os.path.join(data_dir, 'hospitals_london.txt')
nhs_ae_gender_codes = os.path.join(data_dir, 'nhs_ae_gender_codes.csv')
nhs_ae_treatment_codes = os.path.join(data_dir, 'nhs_ae_treatment_codes.csv')
age_population_london = os.path.join(data_dir, 'age_population_london.csv')

