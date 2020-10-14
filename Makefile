## --- echo commands (for debugging)
## SHELL = sh -xv


##-------------------------------------
## Set up main path variables
##-------------------------------------

## set basic variables
PYTHON = python
QUIPP_ROOT = $(shell pwd)

## construct lists of run input .json files and their base prefixes
RUN_INPUTS = $(wildcard run-inputs/*.json)
RUN_INPUTS_BASE_PREFIX = $(patsubst %.json,%,$(notdir $(RUN_INPUTS)))

## construct list of synthetic output directories using the input file names
SYNTH_OUTPUTS_PREFIX = $(addprefix synth-output/,$(RUN_INPUTS_BASE_PREFIX))

## construct list of the synthetic output .csv file names
## (only the 1st synthetic .csv is used)
SYNTH_OUTPUTS_CSV = $(addsuffix /synthetic_data_1.csv,$(SYNTH_OUTPUTS_PREFIX))

## Construct a list of .json file names for each utility and privacy metric
SYNTH_OUTPUTS_PRIV_DISCL_RISK = $(addsuffix /privacy_disclosure_risk.json,$(SYNTH_OUTPUTS_PREFIX))
SYNTH_OUTPUTS_UTIL_CLASS = $(addsuffix /utility_classifiers.json,$(SYNTH_OUTPUTS_PREFIX))
SYNTH_OUTPUTS_UTIL_CORR = $(addsuffix /utility_correlations.json,$(SYNTH_OUTPUTS_PREFIX))

.PHONY: all all-synthetic generated-data clean

all: $(SYNTH_OUTPUTS_PRIV_DISCL_RISK) $(SYNTH_OUTPUTS_UTIL_CLASS) $(SYNTH_OUTPUTS_UTIL_CORR)

all-synthetic: $(SYNTH_OUTPUTS_CSV)


##-------------------------------------
## Generate input data
##-------------------------------------

## set data file paths
AE_DEIDENTIFIED_DATA = generator-outputs/odi-nhs-ae/hospital_ae_data_deidentify.csv generator-outputs/odi-nhs-ae/hospital_ae_data_deidentify.json
LONDON_POSTCODES = generators/odi-nhs-ae/data/London\ postcodes.csv
generated-data: $(AE_DEIDENTIFIED_DATA)

# download the London Postcodes dataset used by the A&E generated
# dataset (this is about 133 MB)
$(LONDON_POSTCODES):
	cd generators/odi-nhs-ae/ && \
	curl -o "./data/London postcodes.csv" \
		https://www.doogal.co.uk/UKPostcodesCSV.ashx?region=E12000007

# make the "A&E deidentified" generated dataset
# this is currently the only generated dataset, so it is handled with
# its own rule
$(AE_DEIDENTIFIED_DATA) &: $(LONDON_POSTCODES)
	mkdir -p generator-outputs/odi-nhs-ae/ && \
	cd generator-outputs/odi-nhs-ae/ && \
	$(PYTHON) $(QUIPP_ROOT)/generators/odi-nhs-ae/generate.py && \
	$(PYTHON) $(QUIPP_ROOT)/generators/odi-nhs-ae/deidentify.py


##-------------------------------------
## Generate synthetic data
##-------------------------------------

## synthesize data - this rule also builds "synth-output/%/data_description.json"
$(SYNTH_OUTPUTS_CSV) : \
synth-output/%/synthetic_data_1.csv : run-inputs/%.json $(AE_DEIDENTIFIED_DATA)
	mkdir -p $$(dirname $@) && \
	python synthesize.py -i $< -o $$(dirname $@)


##-------------------------------------
## Calculate privacy and utility
##-------------------------------------

## compute privacy and utility metrics
$(SYNTH_OUTPUTS_PRIV_DISCL_RISK) : \
synth-output/%/privacy_disclosure_risk.json : \
run-inputs/%.json synth-output/%/synthetic_data_1.csv
	python privacy-metrics/disclosure_risk.py -i $< -o $$(dirname $@)

$(SYNTH_OUTPUTS_UTIL_CLASS) : \
synth-output/%/utility_classifiers.json : \
run-inputs/%.json synth-output/%/synthetic_data_1.csv
	python utility-metrics/classifiers.py -i $< -o $$(dirname $@)

$(SYNTH_OUTPUTS_UTIL_CORR) : \
synth-output/%/utility_correlations.json : \
run-inputs/%.json synth-output/%/synthetic_data_1.csv
	python utility-metrics/correlations.py -i $< -o $$(dirname $@)


##-------------------------------------
## Clean
##-------------------------------------

clean:
	rm -rf generator-outputs synth-output
