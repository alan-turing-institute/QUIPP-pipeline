# Data Synthesis Methods

This directory contains wrappers for the synthesis methods usable in
the pipeline.  Each directory contains a `run` script used by the
pipeline: see [Adding another synthesis
method](../README.md#adding-another-synthesis-method) for the
requirements on this.

 - [ctgan](https://pypi.org/project/ctgan/) (Conditional GAN for Tabular data)
 - [SGF](https://vbinds.ch/node/69) (Synthetic Data Generation Framework)
 - [synthpop](https://CRAN.R-project.org/package=synthpop) (multiple imputations library in R)
 
In addition, there are placeholders and worked examples for other methods
 - Base: a synthesizer Python base class
 - [mice](https://CRAN.R-project.org/package=mice)
 - [SDV](https://pypi.org/project/sdv/) (Synthetic Data Vault,
   parametric model synthesis library in Python, using the
   multivariate version of the Gaussian Copula)
 - [simPop](https://CRAN.R-project.org/package=simPop)
   (micro-simulation using IPF, Simulated Annealing and model-based
   synthesis, designed for datasets with household
   structure). Currently only on embedded Austrian census data set)
