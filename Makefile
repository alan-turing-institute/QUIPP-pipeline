PYTHON = python
QUIPP_ROOT = $(shell pwd)

RUN_INPUTS = $(wildcard run-inputs/*.json)

## construct the synthetic output datafile names
RUN_INPUTS_BASE_PREFIX = $(patsubst %.json,%,$(notdir $(RUN_INPUTS)))
SYNTH_OUTPUTS_PREFIX = $(addprefix synth-output/,$(RUN_INPUTS_BASE_PREFIX))

## the synthetic data is in files synthetic_data_1.csv,
## synthetic_data_2.csv, ...
SYNTH_OUTPUTS_CSV = $(addsuffix /synthetic_data_1.csv,$(SYNTH_OUTPUTS_PREFIX))

## The privacy and utility scores: 
SYNTH_OUTPUTS_DISCL_RISK = $(addsuffix /disclosure_risk.json,$(SYNTH_OUTPUTS_PREFIX))

SYNTH_OUTPUTS_UTIL_SKLEARN = $(addsuffix /sklearn_classifiers.json,$(SYNTH_OUTPUTS_PREFIX))

.PHONY: all all-synthetic clean

all: $(SYNTH_OUTPUTS_DISCL_RISK) $(SYNTH_OUTPUTS_UTIL_SKLEARN)

all-synthetic: $(SYNTH_OUTPUTS_CSV)

## ----------------------------------------
## Generated data

#AE_DEIDENTIFIED_DATA = generator-outputs/odi-nhs-ae/hospital_ae_data_deidentify.csv generator-outputs/odi-nhs-ae/hospital_ae_data_deidentify.json

#LONDON_POSTCODES = generators/odi-nhs-ae/data/London\ postcodes.csv

#generated-data: $(AE_DEIDENTIFIED_DATA)

# Download the London Postcodes dataset used by the A&E generated
# dataset (this is about 133 MB)
#$(LONDON_POSTCODES):
#	cd generators/odi-nhs-ae/ && \
#	curl -o "./data/London postcodes.csv" \
#		https://www.doogal.co.uk/UKPostcodesCSV.ashx?region=E12000007

# Make the "A&E deidentified" generated dataset
#
# This is currently the only generated dataset, so it is handled with
# its own rule
#
#$(AE_DEIDENTIFIED_DATA) &: $(LONDON_POSTCODES)
#	mkdir -p generator-outputs/odi-nhs-ae/ && \
#	cd generator-outputs/odi-nhs-ae/ && \
#	$(PYTHON) $(QUIPP_ROOT)/generators/odi-nhs-ae/generate.py && \
#	$(PYTHON) $(QUIPP_ROOT)/generators/odi-nhs-ae/deidentify.py


## ----------------------------------------
## Synthetic data

## This rule also builds "synth-output/%/data_description.json"
$(SYNTH_OUTPUTS_CSV) : \
synth-output/%/synthetic_data_1.csv : run-inputs/%.json
	mkdir -p $$(dirname $@) && \
	python synthesize.py -i $< -o $$(dirname $@)


## ----------------------------------------
## Privacy and utility metrics
$(SYNTH_OUTPUTS_DISCL_RISK) : \
synth-output/%/disclosure_risk.json : \
run-inputs/%.json synth-output/%/synthetic_data_1.csv
	python privacy-metrics/disclosure_risk.py -i $< -o $$(dirname $@)

$(SYNTH_OUTPUTS_UTIL_SKLEARN) : \
synth-output/%/sklearn_classifiers.json : \
run-inputs/%.json synth-output/%/synthetic_data_1.csv
	python utility-metrics/sklearn_classifiers.py -i $< -o $$(dirname $@)


## ----------------------------------------
## Clean

clean:
	rm -rf synth-output
