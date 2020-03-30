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
```

```
R
> source("env-configuration/install.R")
> q()
```

#### SGF

Another external dependency is the SGF implementation of plausible
deniability:
  
 - Download SGF [here](https://vbinds.ch/node/69)
 - See the library's README file for how to compile the code.  You will
need a recent version of cmake (tested with version 3.17), either
installed through your system's package manager, or from
[here](https://cmake.org/download/).
 - After compilation, the three executables of the SGF package
(`sgfinit`, `sgfgen` and `sgfextract`) should have been built.  Add
their location to your PATH, or alternatively, assign the
environmental variable `SGFROOT` to point to this location.  That is, in bash,
   - either ```export PATH=$PATH:/path/to/sgf/bin```,
   - or ```export SGFROOT=/path/to/sgf/bin```

## Top-level directory contents

The top-level directory structure mirrors the data pipeline.

 - `doc`: The QUiPP report - a high-level overview of the project, our
   work and the methods we have considered so far.

 - `examples`: Tutorial examples of using some of the methods
   (_currently just CTGAN_).  These are independent of the pipeline.

 - `binder`: Configuration files to set up the pipeline using
   [Binder](https://mybinder.org)
 
 - `env-configuration`: Set-up of the computational environment needed
   by the pipeline and its dependencies
 
 - `generators`: Quickly generating input data for the pipeline from a
   few tunable and well-understood models

 - `datasets`: A few (public, open) datasets that we use as input to
   the methods are reproduced here where licence and size permit

 - `synth-methods`: One directory per library/tool, each of them
   implementing a complete synthesis method

 - `utility-metrics`: Scripts relating to computing the utility
   metrics
 
 - `privacy-metrics`: Scripts relating to computing the privacy
   metrics

 - `run-inputs`: Parameter json files (see below), one for each run

When the pipeline is run, additional directories are created:

 - `generator-outputs`: Sample generated input data (using
   `generators`)
   
 - `synth-output`: Contains the result of each run (as specified in
   `run-inputs`), which will typically consist of the synthetic data
   itself and a selection of utility and privacy scores

## The pipeline

The following indicates the full pipeline, as run on an input file
called `example.json`.  This input file has keywords `dataset` (the
base of the filename to use for the original input data) and
`synth-method` which refers to one of the synthesis methods.  As
output, the pipeline produces synthetic data (in one or more files
`synthetic_data_1.csv`, `synthetic_data_2.csv`, ...) and
`metrics.json`, containing the privacy and utility scores.

![Flowchart of the pipeline](doc/fig/pipeline.svg)

The files "dataset.csv" and "dataset.json" could be in
[datasets](datasets/), but this is not a requirement.

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

The input data should be present as two files with the same prefix: a
[csv](https://tools.ietf.org/html/rfc4180.html) file (with suffix
`.csv`) which must contain column headings (along with the column data
itself), and a [json file](doc/schema/) (the "data json file")
describing the types of the columns used for synthesis.

For example, see [the Polish dataset](datasets/polish_data_2011/).
This contains the files

 - `datasets/polish_data_2011/polish_data_2011.csv`
 - `datasets/polish_data_2011/polish_data_2011.json` 
 
and so has the prefix `datasets/polish_data_2011/polish_data_2011`
relative to the root of this repository.

The prefix of the data files (as an absolute path, or relative to the
root of the repository) is given in the parameter json file (see the
next section) as the top-level property `dataset`: there is no
restriction on where these can be located, although a few examples can
be found in `datasets/`.

## Parameter format

The pipeline takes a single json file, describing the data synthesis
to perform, including any parameters the synthesis method might need,
as well as any additional parameters for the privacy and utlity
methods.  The json schema for this parameter file is [here](doc/schema/).

**To be usable by the pipeline, the parameter input file must be
located in the `run-inputs` directory**

