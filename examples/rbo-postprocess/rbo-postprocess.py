import argparse
import json
import matplotlib.pyplot as plt
import subprocess
import numpy as np
import pandas as pd
from itertools import product
from pathlib import Path
from matplotlib import ticker, colors

LABEL_FONT_SIZE = 12

def read_json(fname):
    with open(fname) as f:
        return json.load(f)


def collect_results(path, feature_importance_kind):
    params = read_json((path/path.parts[-1]).with_suffix('.json'))
    
    results_path = path/'utility_feature_importance.json'
    if results_path.exists():
        results = read_json(results_path)
        if feature_importance_kind=='shapley':
            print(results_path)
 
        return {**params['parameters'], **results[feature_importance_kind]} 
    else:
        return {**params['parameters']}


def fig1(data_dir, output_path, feature_importance_kind, privbayes_pat, resampling_pat, include_resampling=True):
    outdir = Path(data_dir)
    privbayes_3 = outdir.glob(privbayes_pat)
    resampling = outdir.glob(resampling_pat)

    pb3 = pd.DataFrame(collect_results(d, feature_importance_kind) for d in privbayes_3)

    if include_resampling:
        rs1 = pd.DataFrame([collect_results(d, feature_importance_kind) for d in resampling])
    else:
        rs1 = []
        
    fig, ax = plt.subplots(1, 3,
                           sharey='row',
                           gridspec_kw={'hspace': 0,
                                        'wspace': 0,
                                        'width_ratios': [1,1,8]},
                           figsize=(15,10))
    
    fig.subplots_adjust(hspace=0)

    ax[0].set_ylim(bottom=0)
    ax[0].set_xlabel('Random feat. permutation', rotation=90, fontsize=LABEL_FONT_SIZE)
    ax[0].set_ylabel("Rank-biased overlap of feature importance", fontsize=LABEL_FONT_SIZE)
    ax[0].tick_params(
        axis='x',
        which='both',
        bottom=False,
        labelbottom=False
    )

    ax[1].set_xlabel('Indep. col. samples', rotation=90, fontsize=LABEL_FONT_SIZE)
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

    
    ####### subplot 0
    if include_resampling:
        rs1.boxplot(
            column="rbo_rand_0.8",
            grid=False,
            widths=0.12,
            sym='.',
            ax=ax[0]
        )
        
    ####### subplot 1
        rs1.boxplot(
            column="rbo_ext_0.8",
            grid=False,
            widths=0.12,
            sym='.',
            ax=ax[1]
        )

    # ###### subplot 3
    pb3.boxplot(
        by="epsilon",
        column="rbo_0.8",
        positions=np.log(sorted(pb3['epsilon'].unique())),
        grid=False,
        widths=0.2,
        sym='.',
        ax=ax[2]
    )

    ax[2].set_xlabel('ε', fontsize=LABEL_FONT_SIZE+3)
    ax[2].tick_params(
        axis='y',
        which='both',
        left=False,
        labelbottom=False
    )
    ax[2].set_title('PrivBayes', y=-0.14)
 
    fig.suptitle('')

    fig.tight_layout()
    fig.savefig(output_path)
    

def fig2(data_dir, output_path, feature_importance_kind, subsampling_pat):
    outdir = Path(data_dir)
    adult_subsamples = outdir.glob(subsampling_pat)

    ss1 = pd.DataFrame(collect_results(d, feature_importance_kind) for d in adult_subsamples)

    fig = plt.figure()
    ax = fig.subplots()

    ss1.boxplot(
        by='frac_samples_to_synthesize',
        column='rbo_0.8',
        positions=np.log(sorted(ss1['frac_samples_to_synthesize'].unique())),
        grid=False,
        widths=0.2,
        sym='.',
        ax=ax
    )

    ax.set_xlabel('sample fraction', fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel('Rank-biased overlap of feature importance', fontsize=LABEL_FONT_SIZE)
    ax.set_title('Sampling without replacement')

    ax.set_ylim((0.0,1.0))
    
    fig.suptitle('')

    fig.tight_layout()

    fig.savefig(output_path)


def fig_l2_rbd_scatter(data_dir, output_path, feature_importance_kind):
    outdir = Path(data_dir)
    privbayes_3 = outdir.glob('privbayes-adult-ensemble-3-*/')
    pb3 = pd.DataFrame(collect_results(d, feature_importance_kind) for d in privbayes_3)
    pb3['rbd_ext_0.8'] = 1.0 - pb3['rbo_ext_0.8']

    fig = plt.figure()
    ax = fig.subplots()

    s = ax.scatter(x=pb3['rbd_ext_0.8'], y=pb3['l2_norm'], c=pb3['epsilon'],
                   cmap=plt.cm.PuBu_r,
                   norm=colors.LogNorm(vmin=0.0001, vmax=40.0))
    
    ax.set_title('l2 norm vs RBD (RBD = 1 - RBO), Privbayes, Adult dataset')
    ax.set_xlabel('RBD (p=0.8)', fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel('L2 dist. feature importance', fontsize=LABEL_FONT_SIZE)
    fig.colorbar(s).set_label('ε', fontsize=LABEL_FONT_SIZE)
    
    fig.savefig(output_path)
    

def main():
    ### Adult dataset
    
    fig1('../../synth-output', 'rbo-builtin-privbayes-adult.pdf', 'builtin',
         privbayes_pat='privbayes-adult-ensemble-shap-*/',
         resampling_pat='adult-resampling-ensemble-*/')

    fig1('../../synth-output', 'rbo-permutation-privbayes-adult.pdf', 'permutation',
         privbayes_pat='privbayes-adult-ensemble-shap-*/',
         resampling_pat='adult-resampling-ensemble-*/')

    fig1('../../synth-output', 'rbo-shapley-privbayes-adult.pdf', 'shapley',
         privbayes_pat='privbayes-adult-ensemble-shap-*/',
         resampling_pat=None,
         include_resampling=False)
    
    fig2('../../synth-output', 'rbo-builtin-subsamples-adult.pdf', 'builtin',
         subsampling_pat='adult-subsample-ensemble-*/')
    fig2('../../synth-output', 'rbo-permutation-subsamples-adult.pdf', 'permutation',
         subsampling_pat='adult-subsample-ensemble-*/')
    
    fig_l2_rbd_scatter('../../synth-output', 'l2_rbd-permutation-privbayes-adult.pdf', 'permutation')

    ### Framingham dataset

    fig1('../../synth-output', 'rbo-permutation-privbayes-framingham.pdf', 'permutation',
         privbayes_pat='privbayes-framingham-ensemble-*/',
         resampling_pat='adult-resampling-ensemble-*/',
         include_resampling=False)

    fig2('../../synth-output', 'rbo-permutation-subsample-framingham.pdf', 'permutation',
         subsampling_pat='subsample-framingham-ensemble-*/')

    
if __name__ == '__main__':
    main()

