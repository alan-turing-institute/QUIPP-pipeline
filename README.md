# QUIPP-pipeline
Privacy preserving synthetic data generation workflows

Collaboration and project management is in the
[QUIPP-collab](https://github.com/alan-turing-institute/QUIPP-collab)
repo (which is currently private).

This repository contains a pipeline for synthetic population
generation, using a variety of methods as implemented by several
libraries.  Measures of privacy and utility can be built on this.

### Top-level directory contents

 - `generators`: Quickly generating input data for the pipeline from a
   few tunable and well-understood models

 - `datasets`: A few (public, open) datasets that we use as input to
   the methods are reproduced here where licence and size permit

 - `synth-methods`: One directory per library/tool, implementing a
   complete synthesis.

 - `run-inputs`: Parameter json files (see below) for each run.


## Running the pipeline

1. Make a parameter json file, in `run-inputs/`, for each desired
   synthesis (see below for the structure of these files).

2. Run `make` in the top level QUIPP-pipeline directory to run all
   syntheses (one per file).  The output for `run-inputs/example.json`
   can be found in `synth-output/example/`.

3. `make clean` removes all synthetic output and generated data.


## Adding another synthesis method

1. Make a subdirectory in `synth-methods` having the name of the new
   method.

2. This directory should contain an executable file `run` that when
   called as

   ```bash
   run $input_json $dataset_base $outfile_prefix
   ```

   runs the method with the input parameter json file on the dataset
   `$dataset_base.{csv,json}` (see data format, below), and puts its
   output files in the directory `$outfile_prefix`.

3. In the parameter json file (e.g. in `run-inputs`), the method can
   be used as the value of the `"synth-method"` name.


## Data format

TODO: describe requirements on csv and json (data) files

## Parameter format

TODO: parameter json files
