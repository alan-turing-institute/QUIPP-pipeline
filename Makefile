PYTHON = python
QUIPP_ROOT = $(shell pwd)

RUN_INPUTS = $(wildcard run-inputs/*.json)

## construct the synthetic output datafile names
RUN_INPUTS_BASE_PREFIX = $(patsubst %.json,%,$(notdir $(RUN_INPUTS)))
SYNTH_OUTPUTS_PREFIX = $(addprefix synth-output/,$(RUN_INPUTS_BASE_PREFIX))
SYNTH_OUTPUTS_CSV = $(addsuffix /synthetic_data_1.csv,$(SYNTH_OUTPUTS_PREFIX))

.PHONY: all-synthetic generated-data clean

all-synthetic: $(SYNTH_OUTPUTS_CSV)

## ----------------------------------------
## Generated data

AE_DEIDENTIFIED_DATA = generator-outputs/odi-nhs-ae/hospital_ae_data_deidentify.csv generator-outputs/odi-nhs-ae/hospital_ae_data_deidentify.json

LONDON_POSTCODES = generators/odi-nhs-ae/data/London\ postcodes.csv

generated-data: $(AE_DEIDENTIFIED_DATA)

# Download the London Postcodes dataset used by the A&E generated dataset
$(LONDON_POSTCODES):
	cd generators/odi-nhs-ae/ && \
	curl -o "./data/London postcodes.csv" \
		https://www.doogal.co.uk/UKPostcodesCSV.ashx?region=E12000007

# Make the "A&E deidentified" generated dataset
$(AE_DEIDENTIFIED_DATA) &: $(LONDON_POSTCODES)
	mkdir -p generator-outputs/odi-nhs-ae/ && \
	cd generator-outputs/odi-nhs-ae/ && \
	$(PYTHON) $(QUIPP_ROOT)/generators/odi-nhs-ae/generate.py && \
	$(PYTHON) $(QUIPP_ROOT)/generators/odi-nhs-ae/deidentify.py


## ----------------------------------------
## Synthetic data

## This rule also builds "synth-output/%/data_description.json"
$(SYNTH_OUTPUTS_CSV) : \
synth-output/%/synthetic_data_1.csv : run-inputs/%.json $(AE_DEIDENTIFIED_DATA)
	mkdir -p $$(dirname $@) && \
	python synthesize.py -i $< -o $$(dirname $@)
	python privacy-metrics/disclosure_risk.py -i $< -o $$(dirname $@)
	python utility-metrics/sklearn_classifiers.py -i $< -o $$(dirname $@)


## ----------------------------------------
## Clean

clean:
	rm -rf generator-outputs synth-output
