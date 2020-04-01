# SGF

[SGF](https://vbinds.ch/node/69) is a synthetic generation tool that provides
a privacy guarantee based on plausible deniability.

See also [this paper](https://arxiv.org/abs/1708.07975).

## Additional synthesis parameters

These directly correspond to configuration options passed to the sgf tools.
More detail about these inputs can be found in the SGF documentation.

- `gamma` (_number_): Must be > 1.0.  Privacy parameter which controls how the
  *plausible seeds* for a record are determined.
- `omega` (_integer_): The number of resampled attributes is N - omega
- `ncomp` (_string_): 
- `ndist` (_string_): The noise distribution in the generative model: "none", 
  "lap" (Laplacian) or "geom" (Geometric).
- `k` (_integer_): 
- `epsilon0` (_number_):
- `tinc` (_integer_):

### Example
[sgf-example-1.json](../../run-inputs/sgf-example-1.json)
