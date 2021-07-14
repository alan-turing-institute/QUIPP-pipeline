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
SYNTH_OUTPUTS_PRIV_DISCL_RISK = $(addsuffix /disclosure_risk.json,$(SYNTH_OUTPUTS_PREFIX))
SYNTH_OUTPUTS_UTIL_CLASS = $(addsuffix /utility_diff.json,$(SYNTH_OUTPUTS_PREFIX))
SYNTH_OUTPUTS_UTIL_CORR = $(addsuffix /utility_correlations.json,$(SYNTH_OUTPUTS_PREFIX))
SYNTH_OUTPUTS_UTIL_FEATURE_IMPORTANCE = $(addsuffix /utility_feature_importance.json,$(SYNTH_OUTPUTS_PREFIX))

.PHONY: all all-synthetic generated-data clean

all: $(SYNTH_OUTPUTS_PRIV_DISCL_RISK) $(SYNTH_OUTPUTS_UTIL_CLASS) $(SYNTH_OUTPUTS_UTIL_CORR) $(SYNTH_OUTPUTS_UTIL_FEATURE_IMPORTANCE)

all-synthetic: $(SYNTH_OUTPUTS_CSV)


##-------------------------------------
## Add provenance information to output
##-------------------------------------

PROVENANCE_DEF = provenance() {\
    git_result=$$(python provenance.py | jq '{commit, local_modifications}') ; \
    ( jq ". += {git: $$git_result}" $$1 > $${1}.tmp ) && mv $${1}.tmp $$1 || \
    echo "Warning: No provenance could be recorded for this target" && rm -f $${1}.tmp ; \
}
ADD_PROVENANCE = $(PROVENANCE_DEF) && provenance


##-------------------------------------
## Generate input data
##-------------------------------------

## set data file paths
AE_DEIDENTIFIED_DATA = generator-outputs/odi-nhs-ae/hospital_ae_data_deidentify.csv generator-outputs/odi-nhs-ae/hospital_ae_data_deidentify.json
LONDON_POSTCODES = generators/odi-nhs-ae/data/London\ postcodes.csv
HP_DATA_CLEAN = generator-outputs/household_poverty/train_cleaned.csv generator-outputs/household_poverty/train_cleaned.json
ARTIFICIAL_DATA_1 = generator-outputs/artificial/artificial_1.csv generator-outputs/artificial/artificial_1.json
ARTIFICIAL_DATA_2 = generator-outputs/artificial/artificial_2.csv generator-outputs/artificial/artificial_2.json
ARTIFICIAL_DATA_3 = generator-outputs/artificial/artificial_3.csv generator-outputs/artificial/artificial_3.json
ARTIFICIAL_DATA_4 = generator-outputs/artificial/artificial_4.csv generator-outputs/artificial/artificial_4.json
ARTIFICIAL_DATA_5 = generator-outputs/artificial/artificial_5.csv generator-outputs/artificial/artificial_5.json
ARTIFICIAL_DATA_6 = generator-outputs/artificial/artificial_6.csv generator-outputs/artificial/artificial_6.json
ARTIFICIAL_DATA_7 = generator-outputs/artificial/artificial_7.csv generator-outputs/artificial/artificial_7.json
generated-data: $(AE_DEIDENTIFIED_DATA) $(HP_DATA_CLEAN) $(ARTIFICIAL_DATA_1) $(ARTIFICIAL_DATA_2) $(ARTIFICIAL_DATA_3) $(ARTIFICIAL_DATA_4) $(ARTIFICIAL_DATA_5) $(ARTIFICIAL_DATA_6) $(ARTIFICIAL_DATA_7)

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
    mkdir -p generator-outputs/household_poverty/ && \
	cd generator-outputs/odi-nhs-ae/ && \
	$(PYTHON) $(QUIPP_ROOT)/generators/odi-nhs-ae/generate.py && \
	$(PYTHON) $(QUIPP_ROOT)/generators/odi-nhs-ae/deidentify.py && \
	cd ../household_poverty/ && \
	$(PYTHON) $(QUIPP_ROOT)/generators/household_poverty/clean.py

# pre-process the Household Poverty dataset
$(HP_DATA_CLEAN):
	mkdir -p generator-outputs/household_poverty/ && \
	cd generator-outputs/household_poverty/ && \
	$(PYTHON) $(QUIPP_ROOT)/generators/household_poverty/clean.py

# generate the three artificial datasets
$(ARTIFICIAL_DATA_1) $(ARTIFICIAL_DATA_2) $(ARTIFICIAL_DATA_3)  $(ARTIFICIAL_DATA_4)  $(ARTIFICIAL_DATA_5)  $(ARTIFICIAL_DATA_6)  $(ARTIFICIAL_DATA_7):
	mkdir -p generator-outputs/artificial/ && \
	cd generator-outputs/artificial/ && \
	$(PYTHON) $(QUIPP_ROOT)/generators/artificial/generate.py


##-------------------------------------
## Generate synthetic data
##-------------------------------------

## synthesize data - this rule also builds "synth-output/%/data_description.json"
$(SYNTH_OUTPUTS_CSV) : \
synth-output/%/synthetic_data_1.csv : run-inputs/%.json $(AE_DEIDENTIFIED_DATA) $(HP_DATA_CLEAN) $(ARTIFICIAL_DATA_1) $(ARTIFICIAL_DATA_2) $(ARTIFICIAL_DATA_3) $(ARTIFICIAL_DATA_4) $(ARTIFICIAL_DATA_5) $(ARTIFICIAL_DATA_6) $(ARTIFICIAL_DATA_7)
	outdir=$$(dirname $@) && \
	mkdir -p $$outdir && \
	cp $< $${outdir}/input.json && \
	$(ADD_PROVENANCE) $${outdir}/input.json && \
	python synthesize.py -i $< -o $$outdir


##-------------------------------------
## Calculate privacy and utility
##-------------------------------------

## compute privacy and utility metrics
$(SYNTH_OUTPUTS_PRIV_DISCL_RISK) : \
synth-output/%/disclosure_risk.json : \
run-inputs/%.json synth-output/%/synthetic_data_1.csv
	python metrics/privacy-metrics/disclosure_risk.py -i $< -o $$(dirname $@) &&\
	$(ADD_PROVENANCE) $@

$(SYNTH_OUTPUTS_UTIL_CLASS) : \
synth-output/%/utility_diff.json : \
run-inputs/%.json synth-output/%/synthetic_data_1.csv
	python metrics/utility-metrics/classifiers.py -i $< -o $$(dirname $@) &&\
	$(ADD_PROVENANCE) $@

$(SYNTH_OUTPUTS_UTIL_CORR) : \
synth-output/%/utility_correlations.json : \
run-inputs/%.json synth-output/%/synthetic_data_1.csv
	python metrics/utility-metrics/correlations.py -i $< -o $$(dirname $@) &&\
	$(ADD_PROVENANCE) $@

$(SYNTH_OUTPUTS_UTIL_FEATURE_IMPORTANCE) : \
synth-output/%/utility_feature_importance.json : \
run-inputs/%.json synth-output/%/synthetic_data_1.csv
	python metrics/utility-metrics/feature_importance.py -i $< -o $$(dirname $@) &&\
	$(ADD_PROVENANCE) $@


##-------------------------------------
## Helper targets for individual inputs
##-------------------------------------

##   make run-example
##
## produces synthetic data and metrics from run-inputs/example.json with output in synth-output/example/

run-% :\
synth-output/%/synthetic_data_1.csv\
synth-output/%/utility_correlations.json\
synth-output/%/disclosure_risk.json\
synth-output/%/utility_diff.json\
synth-output/%/utility_feature_importance.json\
;


##-------------------------------------
## Clean
##-------------------------------------

clean:
	rm -rf generator-outputs synth-output
