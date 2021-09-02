"""A 'privacy metric' that produce a copy of `synthetic_data_1.csv`,
but with a fraction of records substituted by records from the
individual dataset.

The `disclosure_risk.py` is run on each output with the same
parameters, providing a test that this method performs as expected.

WARNING: Given that they contain records from the original dataset,
these data sets may not be disclosive.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import json

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, "utilities"))
from utils import handle_cmdline_args


def combine_fractional(df1, df2, p, seed=1234):
    """Returns a dataframe the same shape as df1, where a fraction `p`
of the rows chosen uniformly at random from df2 with replacement."""

    rng = np.random.default_rng(seed=seed)

    df2_resampled = df2.sample(len(df1), replace=True).reset_index(drop=True)

    r_rows = rng.choice([True, False], df1.shape[0], p=(p, 1 - p))
    r_elts = np.stack((r_rows,) * df1.shape[1], axis=-1)

    return df1.mask(r_elts, df2_resampled, axis=0)


def main():
    args = handle_cmdline_args()

    with open(args.infile) as f:
        synth_params = json.load(f)

    orig_data_file = Path(synth_params["dataset"]).with_suffix(".csv")
    orig_data = pd.read_csv(orig_data_file)

    rlsd_data_file = args.outfile_prefix / Path("synthetic_data_1.csv")
    rlsd_data = pd.read_csv(rlsd_data_file)

    for p in [0.01, 0.02, 0.05]:
        leaky_output = combine_fractional(orig_data, rlsd_data, 0.01)
        leaky_output.to_csv(
            Path(args.outfile_prefix)
            / Path("synth_data_leaked_" + str(round(p * 100))).with_suffix(".csv")
        )


if __name__ == "__main__":
    main()
