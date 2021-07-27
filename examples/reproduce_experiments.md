# Reproduce experiments

This document explains how to reproduce all experimental runs and plots.

## Requirements
The easiest way to use the pipeline is with [Docker](https://www.docker.com/). Get the official image

```bash
docker pull turinginst/quipp-env:base
```

# Experiments

Check out the [QUIPP-pipeline](https://github.com/alan-turing-institute/QUIPP-pipeline) repository and set as the current directory

```bash
git clone https://github.com/alan-turing-institute/QUIPP-pipeline.git && cd QUIPP-pipeline
```

All experiments can then be run with the docker image and all results will be saved to `/synth-output`, which is mounted to the docker container.


## Artificial 1

```bash
docker run -v $(pwd):/quipp-pipeline --workdir /quipp-pipeline turinginst/quipp-env:base python examples/privbayes-artificial_1-ensemble/privbayes-artificial_1-ensemble.py -n 25 -f -k 3 -e 0.0001,0.001,0.01,0.1,0.4,1.0,4.0,10.0 -r 
```

### Artificial 2

```bash
docker run -v $(pwd):/quipp-pipeline --workdir /quipp-pipeline turinginst/quipp-env:base python examples/privbayes-artificial_2-ensemble/privbayes-artificial_2-ensemble.py -n 25 -f -k 3 -e 0.0001,0.001,0.01,0.1,0.4,1.0,4.0,10.0 -r 
```

### Artificial 3

```bash
docker run -v $(pwd):/quipp-pipeline --workdir /quipp-pipeline turinginst/quipp-env:base python examples/privbayes-artificial_3-ensemble/privbayes-artificial_3-ensemble.py -n 25 -f -k 3 -e 0.0001,0.001,0.01,0.1,0.4,1.0,4.0,10.0 -r 
```

### Artificial 4

```bash
docker run -v $(pwd):/quipp-pipeline --workdir /quipp-pipeline turinginst/quipp-env:base python examples/privbayes-artificial_4-ensemble/privbayes-artificial_4-ensemble.py -n 25 -f -k 3 -e 0.0001,0.001,0.01,0.1,0.4,1.0,4.0,10.0 -r 
```

### Adult

**PrivBayes**
```bash
docker run -v $(pwd):/quipp-pipeline --workdir /quipp-pipeline turinginst/quipp-env:base python examples/privbayes-adult-ensemble/privbayes-adult-ensemble.py -n 25 -k 3 -e 0.0001,0.001,0.01,0.1,0.4,1.0,4.0,10.0  -f -r 
```

**Resampling**
```bash
docker run -v $(pwd):/quipp-pipeline --workdir /quipp-pipeline turinginst/quipp-env:base python examples/adult-resampling-ensemble/adult-resampling-ensemble.py -n 50 -f -r 
```

**Subsample**
```bash
docker run -v $(pwd):/quipp-pipeline --workdir /quipp-pipeline turinginst/quipp-env:base python examples/adult-subsample-ensemble/adult-subsample-ensemble.py -n 25 -f -r 
```

### Household Poverty
You must download the Household poverty data following the instructions [here](
generators/household_poverty/data/README.md).

Then preprocess the data:

```bash
docker run -v $(pwd):/quipp-pipeline --workdir /quipp-pipeline turinginst/quipp-env:base make generated-data                                                                    
```

**PrivBayes**
```bash
docker run -v $(pwd):/quipp-pipeline --workdir /quipp-pipeline turinginst/quipp-env:base python examples/privbayes-household-ensemble/privbayes-household-ensemble.py -n 25 -k 3 -e 0.0001,0.001,0.01,0.1,0.4,1.0,4.0,10.0  -f -r 
```

**Resampling**
```bash
docker run -v $(pwd):/quipp-pipeline --workdir /quipp-pipeline turinginst/quipp-env:base python examples/household-resampling-ensemble/household-resampling-ensemble.py -n 50 -f -r 
```

**Subsample**
```bash
docker run -v $(pwd):/quipp-pipeline --workdir /quipp-pipeline turinginst/quipp-env:base python examples/household-subsample-ensemble/household-subsample-ensemble.py -n 25 -f -r 
```

### Polish Survey


**PrivBayes**
```bash
docker run -v $(pwd):/quipp-pipeline --workdir /quipp-pipeline turinginst/quipp-env:base python examples/privbayes-polish-ensemble/privbayes-polish-ensemble.py  -n 25 -k 3 -e 0.0001,0.001,0.01,0.1,0.4,1.0,4.0,10.0  -f -r 
```

**Resampling**
```bash
docker run -v $(pwd):/quipp-pipeline --workdir /quipp-pipeline turinginst/quipp-env:base python examples/polish-resampling-ensemble/polish-resampling-ensemble.py -n 50 -f -r 
```

**Subsample**
```bash
docker run -v $(pwd):/quipp-pipeline --workdir /quipp-pipeline turinginst/quipp-env:base python examples/polish-subsample-ensemble/polish-subsample-ensemble.py -n 25 -f -r 
```