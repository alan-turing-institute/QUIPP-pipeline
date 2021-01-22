import argparse
import json
import matplotlib.pyplot as plt
import subprocess
import numpy as np
import pandas as pd
from itertools import product
from pathlib import Path

def read_json(fname):
    with open(fname) as f:
        return json.load(f)


def collect_results(path):
    params = read_json((path/path.parts[-1]).with_suffix('.json'))
    results = read_json(path/'utility_feature_importance.json')
    return {**params['parameters'], **results}


def main():
    outdir = Path('../../synth-output')
    privbayes_2 = outdir.glob('privbayes-adult-ensemble-2-*/')
    privbayes_3 = outdir.glob('privbayes-adult-ensemble-3-*/')
    resampling = outdir.glob('adult-resampling-ensemble-*/')
    bootstrap = outdir.glob('adult-bootstrap-ensemble-*/')
    
    pb2 = pd.DataFrame(collect_results(d) for d in privbayes_2)
    pb3 = pd.DataFrame(collect_results(d) for d in privbayes_3)    
    privbayes_all = pb2.append(pb3, ignore_index=True).query('epsilon < 100.0')

    rs1 = pd.DataFrame(collect_results(d) for d in resampling)

    bs1 = pd.DataFrame(collect_results(d) for d in bootstrap)

    # rs1.rename(columns={'rbo_rand_0.8': 'Random feature permutation',
    #                     'rbo_0.8': 'Indep. sampling'},
    #            inplace=True)
    
    # pltdata = privbayes_all.groupby('epsilon')['rbo_0.8'].apply(list).reset_index()

    #fig = plt.figure(figsize=(12,5))
    #gs = fig.add_gridspec(1, 2, hspace=0, wspace=0)
    fig, ax = plt.subplots(1, 4,
                           sharey='row',
                           gridspec_kw={'hspace': 0,
                                        'wspace': 0,
                                        'width_ratios': [1,1,1,8]},
                           figsize=(16.5,10))

    label_font_size = 12
    
    fig.subplots_adjust(hspace=0)

    ax[0].set_ylim(bottom=0)
    
    ####### subplot 0
    rs1.boxplot(
        column="rbo_rand_0.8",
        grid=False,
        widths=0.2,
        sym='.',
        ax=ax[0]
    )
    ax[0].set_xlabel('Random feat. permutation', rotation=90, fontsize=label_font_size)
    ax[0].set_ylabel("Rank-biased overlap of feature importance", fontsize=label_font_size)
    ax[0].tick_params(
        axis='x',
        which='both',
        bottom=False,
        labelbottom=False
    )
    
    ####### subplot 1
    rs1.boxplot(
        column="rbo_0.8",
        grid=False,
        widths=0.2,
        sym='.',
        ax=ax[1]
    )
    ax[1].set_xlabel('Indep. col. samples', rotation=90, fontsize=label_font_size)
    ax[1].tick_params(
        axis='y',
        which='both',
        left=False,
        right=False,
        labelbottom=False
    )
    ax[1].tick_params(
        axis='x',
        which='both',
        bottom=False,
        labelbottom=False
    )

    ####### subplot 2
    bs1.boxplot(
        column="rbo_0.8",
        grid=False,
        widths=0.2,
        sym='.',
        ax=ax[2]
    )
    ax[2].set_xlabel('Bootstrap samples', rotation=90, fontsize=label_font_size)
    ax[2].tick_params(
        axis='y',
        which='both',
        left=False,
        right=False,
        labelbottom=False
    )
    ax[2].tick_params(
        axis='x',
        which='both',
        bottom=False,
        labelbottom=False
    )


    
    ###### subplot 3
    privbayes_all.boxplot(
        by="epsilon",
        column="rbo_0.8",
        positions=np.log(sorted(privbayes_all['epsilon'].unique())),
        grid=False,
        widths=0.2,
        sym='.',
        ax=ax[3]
    )
    ax[3].set_xlabel('ε', fontsize=label_font_size+3)
    ax[3].tick_params(
        axis='y',
        which='both',
        left=False,
        labelbottom=False
    )
    ax[3].set_title('PrivBayes', y=-0.14)
    
    #    ax[0].label_outer()
    #    ax[1].label_outer()
    
    #fig.suptitle('Random Forest Feature Importance')
    fig.suptitle('')

    fig.tight_layout()
    fig.savefig("rbo.pdf")
    
    # plt.violinplot(np.array(pltdata['rbo_0.8']), np.log(np.array(pltdata['epsilon'])), showextrema=False)
    # plt.ylabel("RBO(feature importances)")
    # plt.tight_layout()
    # plt.savefig("rbo.pdf")


if __name__ == '__main__':
    main()

