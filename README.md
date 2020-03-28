# QUIPP-pipeline
Privacy preserving synthetic data generation workflows

_Collaboration and project management is in the
[QUIPP-collab](https://github.com/alan-turing-institute/QUIPP-collab)
repo (which is currently private)._

The QUiPP (Quantifying Utility and Preserving Privacy) project aims to
produce a framework to facilitate the creation of synthetic population
data where the privacy of individuals is quantified. In addition,
QUiPP can assess utility in a variety of contexts.  Does a model
trained on the synthetic data generalize as well to the population as
the same model trained on the confidential sample, for example?

The proliferation of individual-level data sets has opened up new
research opportunities.  This individual information is tightly
restricted in many contexts: in health and census records, for example.
This creates difficulties in working openly and reproducibly, since
full analyses cannot then be shared.  Methods exist for creating
synthetic populations that are representative of the existing
relationships and attributes in the original data.  However,
understanding the utility of the synthetic data and simultaneously
protecting individuals' privacy, such that these data can be released
more openly, is challenging.

This repository contains a pipeline for synthetic population
generation, using a variety of methods as implemented by several
libraries.  In addition, the pipeline emits measures of privacy and
utility of the resulting data.

The current draft QUiPP report can be found in `doc`, with a pdf
available
[here](https://github.com/alan-turing-institute/QUIPP-pipeline/releases).

## Installation
- Clone the repository `git clone
  git@github.com:alan-turing-institute/QUIPP-pipeline.git`

## Dependencies

_Note that a Docker image is provided with the dependencies
pre-installed, as
[turinginst/quipp-env](https://hub.docker.com/repository/docker/turinginst/quipp-env).
More detail on setting this up can be found
[here](env-configuration/README.md)._

- Various parts of this code and its dependencies are written in
  Python, R, C++ and Bash.
- It has been tested with
  - python 3.6
  - R 3.6
  - gcc 9.3
  - bash 3.2
- It depends on the following libraries/tools:
  - Python: [numpy](https://pypi.org/project/numpy/), [pandas](https://pypi.org/project/pandas/), [scikit-learn](https://pypi.org/project/scikit-learn/), [scipy](https://pypi.org/project/scipy/), [ctgan](https://pypi.org/project/ctgan/), [sdv](https://pypi.org/project/sdv/), [simanneal](https://pypi.org/project/simanneal/)
  - R: [simPop](https://CRAN.R-project.org/package=simPop), [synthpop](https://CRAN.R-project.org/package=synthpop), [mice](https://CRAN.R-project.org/package=mice), [dplyr](https://CRAN.R-project.org/package=dplyr), [magrittr](https://CRAN.R-project.org/package=magrittr), [tidyr](https://CRAN.R-project.org/package=tidyr)
  - [SGF](https://vbinds.ch/node/69) (Synthetic Data Generation Framework)

### Installing the dependencies

#### R and Python dependencies

To install all of the python and R dependencies, run the following
commands in a terminal from the root of this repository:

```bash
python -m pip install -r env-configuration/requirements.txt
Rscript env-configuration/install.R
```

#### SGF

Another external dependency is the SGF implementation of plausible
deniability:
  
- download SGF [here](https://vbinds.ch/node/69)
  
See the library's README file for how to compile the code.  You will
need a recent version of cmake (tested with version 3.17), either
installed through your system's package manager, or from
[here](https://cmake.org/download/).  After compilation and once the
three executables (`sgfinit`, `sgfgen` and `sgfextract`) have been
built, either:
 
 - these should either be in your PATH; or
 - assign the environmental variable `SGFROOT` to point to the
directory containing these executables, in bash as:
```
export SGFROOT=path/to/executables`
```

## Top-level directory contents

The top-level directory structure mirrors the data pipeline. (TODO more)

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
