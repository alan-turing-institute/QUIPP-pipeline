# Synthpop

[Synthpop](https://CRAN.R-project.org/package=synthpop) is an R package for producing synthetic
microdata using sequential modelling.

See also the [accompanying paper](https://www.jstatsoft.org/article/view/v074i11).

## Additional synthesis parameters

- `vars_sequence` (_array_): the order in which to synthesise the variable (i.e. columns). A variable is synthesised conditional on all the variables that precede it in the list, e.g. (3,5,1) means that variable 1 is synthesised conditional on variables 3 and 5. Note that only variables that have a synthesis method assigned to them in synthesis_methods are actually synthesised. The other variables in vars_sequence are just replicated from the original data.
  - items (_integer_): one-indexed column id
- `synthesis_methods` (_array_): The method to use to synthesize each variable (i.e. column)
  - items (_string_): See the synthpop manual for a complete list of options.  Common
    choices are "sample" (independently resample this column), "" (no synthesis, just replication of the original data), "cart" (synthesize using the Classification And Regression Tree method).  The column created first (i.e. the first in vars_sequence) must contain "sample" in synthesis_methods.
- `proper` (_boolean_): If "true", synthesis is done using the full joint posterior, if "false" it is done sequentially.
- `tree_minbucket` (_integer_): for the tree-based synthesis, the minimum acceptable
  number of real records in each leaf node.
- `smoothing` (_object_): Not implemented at the moment, set to [].

### Example

[synthpop-example-1.json](../../run-inputs/synthpop-example-1.json)
