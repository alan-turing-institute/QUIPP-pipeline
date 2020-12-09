Reproduce Figures of Experiments 6 and 7 [in this issue](https://github.com/alan-turing-institute/QUIPP-collab/issues/120).

**Experiment 6**

* Synthetic method: synthpop
* Dataset: `datasets/polish_data_2011/polish_data_2011`
* random_state: 12345, 23451, 34512
* Columns (original data): 

```
1: "smoke",
2: "sex",
3: "age",
4: "edu",
5: "weight",
6: "height",
7: "bmi",
8: "sport",
9: "marital",
10: "region",
11: "wkabint",
12: "income",
13: "ls"
```

* vars_sequence: [3, 6, 5, 7, 2]

```
Other possible scenarios that we discussed (not used here):

age ---> edu ---> height ---> weight ---> bmi ---> sex 
age ---> height ---> weight ---> sex ---> marital ---> income ---> edu
age ---> marital ---> edu ---> income 
weight ---> bmi ---> sex ---> height
```

* num_samples_intruder: 4895
* vars_intruder: ["sex","age","edu","weight","height","bmi"]
* Utility: 
    - "input_columns": ["age", "edu", "height", "weight", "bmi"],
   - "label_column": "sex",

---

Six synthetic datasets were generated (for each random seed) which only differ in `synthesis_method`:

1. "synthesis_methods": ["", "sample", "sample", "", "sample", "sample", "sample", "", "", "", "", "", ""],
2. "synthesis_methods": ["", "cart", "", "", "cart", "cart", "cart", "", "", "", "", "", ""],
3. "synthesis_methods": ["", "cart", "", "", "cart", "", "cart", "", "", "", "", "", ""],
4. "synthesis_methods": ["", "cart", "", "", "", "", "cart", "", "", "", "", "", ""],
5. "synthesis_methods": ["", "cart", "", "", "", "", "", "", "", "", "", "", ""],
6. "synthesis_methods": ["", "", "", "", "", "", "", "", "", "", "", "", ""],

From 1 (randomly sampled) ---> 6 (original dataset), the synthetic dataset gradually becomes more similar to the original dataset.

Results:

![1](https://user-images.githubusercontent.com/1899856/97453158-60038c00-192d-11eb-9aad-eaa8e28169c3.png)

**Experiment 7**

Comparison between synthpop and CTGAN results for `datasets/polish_data_2011/polish_data_2011`:

![polish](https://user-images.githubusercontent.com/1899856/97554215-59792100-19ce-11eb-8945-b5e099a001cb.png)
