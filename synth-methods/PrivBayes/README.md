# PrivBayes

**This code uses the PrivBayes implementation within the DataSynthesizer fork found here: https://github.com/gmingas/DataSynthesizer. The user needs to pre-install the package using pip locally: 
Clone the above fork, go to its root directory and run `pip install .`**

References:
```tex
@article{10.1145/3134428,
author = {Zhang, Jun and Cormode, Graham and Procopiuc, Cecilia M. and Srivastava, Divesh and Xiao, Xiaokui},
title = {PrivBayes: Private Data Release via Bayesian Networks},
year = {2017},
issue_date = {November 2017},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {42},
number = {4},
issn = {0362-5915},
url = {https://doi.org/10.1145/3134428},
doi = {10.1145/3134428},
journal = {ACM Trans. Database Syst.},
month = oct,
articleno = {25},
numpages = {41},
keywords = {Differential privacy, synthetic data generation, bayesian network}
}

@inproceedings{inproceedings,
author = {Ping, Haoyue and Stoyanovich, Julia and Howe, Bill},
year = {2017},
month = {06},
pages = {1-5},
title = {DataSynthesizer: Privacy-Preserving Synthetic Datasets},
doi = {10.1145/3085504.3091117}
}
```

## Additional synthesis parameters
`category_threshold` (_integer_): maximum number of distinct values for categorical features 
`epsilon` (_float_): epsilon for embedded differential privacy mechanism 
`k` (_integer_): maximum number of parents for each node in the Bayesian network
`keys` (_dict[string, bool]_): features that are treated as table keys 
`histogram_bins` (_integer_): maximum number of bins when converting continuous features to discrete (PrivBayes is not directly compatible with continuous features)
`preconfigured_bn` (dictionary): a list of all the variables in the dataset and their parents in the Bayesian network graph. If set to {}, PrivBayes infers a BN graph
using the greedy bayes algorithm.
        
### Examples
[privbayes-example-0.json](../../run-inputs/privbayes-example-0.json)
[privbayes-example-1.json](../../run-inputs/privbayes-example-0.json)
[privbayes-example-2.json](../../run-inputs/privbayes-example-0.json)
