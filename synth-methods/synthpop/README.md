# Synthpop

[Synthpop](https://CRAN.R-project.org/package=synthpop) is an R package for producing synthetic
microdata using sequential modelling.

See also the [accompanying paper](https://www.jstatsoft.org/article/view/v074i11).

## Additional synthesis parameters

- `vars_sequence` (_array_): the order in which to synthesize columns
  - items (_integer_): one-indexed column id
- `synthesis_methods` (_array_): The method to use to synthesize each column
  - items (_string_): See the synthpop manual for a complete list of options.  Common
    choices are "sample" (independently sample from this column), "" (take field from 
    the most recently sampled column), "cart" (synthesize using the Classification And 
    Regression Tree method).  The column sampled first must contain "sample".
- `proper` (_boolean_): ??
- `tree_minbucket` (_integer_): for the tree-based synthesis, the minimum acceptable
  number of real records in each leaf node.
- `smoothing` (_object_): ??

### Example

[synthpop-example-1.json](../../run-inputs/synthpop-example-1.json)
