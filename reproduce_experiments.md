## Docker
The easiest way to use the pipeline is with [Docker](https://www.docker.com/). Get the official image with

```bash
docker pull turinginst/quipp-env:base
```

To run the pipeline use
```bash
docker run -v $(pwd):/quipp-pipeline --workdir /quipp-pipeline turinginst/quipp-env:base make
```

# Experiments


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
docker run -v $(pwd):/quipp-pipeline --workdir /quipp-pipeline turinginst/quipp-env:base python examples/adult-resampling-ensemble/adult-resampling-ensemble.py  -n 50 -k 3 -e 0.0001,0.001,0.01,0.1,0.4,1.0,4.0,10.0  -f -r 
```

**Subsample**
```bash
docker run -v $(pwd):/quipp-pipeline --workdir /quipp-pipeline turinginst/quipp-env:base python examples/adult-subsample-ensemble/adult-subsample-ensemble.py -n 25 -k 3 -e 0.0001,0.001,0.01,0.1,0.4,1.0,4.0,10.0  -f -r 
```
### Household Poverty

### Polish Survey