import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import mplhep
import os

from utils import *

mplhep.style.use("CMS") 
mplhep.style.use({"savefig.bbox": "tight"})

percentiles = (0.5,98)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--detector', dest='detector', default='EB', help='Part of detector: EB/EEp/EE-')
parser.add_argument('--dataset', dest='dataset', default='test', help='Dataset: test/val/train')
parser.add_argument('--nbins', dest='nbins', default=40, type=int, help='Number of bins in plot')
parser.add_argument('--ext', dest='ext', default="", help='Extension for saving')
parser.add_argument('--do-correction-factor', dest='do_correction_factor', default=False, action="store_true", help='Plot correction factors')
parser.add_argument('--plot-only-S2', dest='plot_only_S2', default=False, action="store_true", help='Plot only shape correction factor')
parser.add_argument('--plot-path', dest='plot_path', default="plots_weight", help='Path to save plots')
args = parser.parse_args()

# Plotting options
nbins = int(args.nbins)

input_mc = pd.read_parquet(f"samples_processed/MC_{args.dataset}_{args.detector}_processed.parquet", engine="pyarrow")

_, ax = plt.subplots()

if args.do_correction_factor:
    lo = np.percentile(input_mc['w_post_S2']/input_mc['w_post_S1'], percentiles[0])
    hi = np.percentile(input_mc['w_post_S2']/input_mc['w_post_S1'], percentiles[1])
else:
    lo = np.percentile(input_mc['w_post_S2'], percentiles[0])
    hi = np.percentile(input_mc['w_post_S2'], percentiles[1])

hists = {}

if args.do_correction_factor:
    hists['kin_rwgt'] = np.histogram(input_mc['w_post_S1']/input_mc['weight_norm'], nbins, (lo,hi))
    hists['shape_rwgt'] = np.histogram(input_mc['w_post_S2']/input_mc['w_post_S1'], nbins, (lo,hi))
    
    # Top panel
    if not args.plot_only_S2:
        mplhep.histplot(
            hists['kin_rwgt'],
            ax = ax,
            histtype = "fill",
            flow = "none",
            color = '#5790fc',
            alpha = 0.1
        )

        mplhep.histplot(
            hists['kin_rwgt'],
            ax = ax,
            label = "Kin rwgt factor (S1)",
            histtype = "step",
            flow = "none",
            color = '#5790fc',
        )

    mplhep.histplot(
        hists['shape_rwgt'],
        ax = ax,
        histtype = "fill",
        flow = "none",
        color = '#e42536',
        alpha = 0.1
    )
    
    mplhep.histplot(
        hists['shape_rwgt'],
        ax = ax,
        label = "Shape rwgt factor (S2)",
        histtype = "step",
        flow = "none",
        color = '#e42536',
    )


else:
    hists['mc_prekin'] = np.histogram(input_mc['weight_norm'], nbins, (lo,hi))
    hists['mc'] = np.histogram(input_mc['w_post_S1'], nbins, (lo,hi))
    hists['mc_rwgt'] = np.histogram(input_mc['w_post_S2'], nbins, (lo,hi))
    
    # Top panel
    mplhep.histplot(
        hists['mc_prekin'],
        ax = ax,
        histtype = "fill",
        flow = "none",
        color = '#a96b59',
        alpha = 0.1
    )
    
    mplhep.histplot(
        hists['mc'],
        ax = ax,
        histtype = "fill",
        flow = "none",
        color = '#5790fc',
        alpha = 0.1
    )
    
    mplhep.histplot(
        hists['mc_rwgt'],
        ax = ax,
        histtype = "fill",
        flow = "none",
        color = '#e42536',
        alpha = 0.1
    )
    
    mplhep.histplot(
        hists['mc_prekin'],
        ax = ax,
        label = "MC (pre kinematic rwgt)",
        histtype = "step",
        flow = "none",
        color = '#a96b59',
    )
    
    mplhep.histplot(
        hists['mc'],
        ax = ax,
        label = "MC",
        histtype = "step",
        flow = "none",
        color = '#5790fc',
    )
    
    mplhep.histplot(
        hists['mc_rwgt'],
        ax = ax,
        label = "MC rwgt",
        histtype = "step",
        flow = "none",
        color = '#e42536',
    )
    
    
ax.set_xlim(lo,hi)
if args.do_correction_factor:
    ax.axvline(1, color='grey', ls='--')

if args.do_correction_factor:
    ax.set_xlabel("Correction factor")
else:
    ax.set_xlabel("Weight")

ax.set_ylabel("Events")

ax.legend(loc='best', fontsize=20)

# Add label
mplhep.cms.label(
    "Preliminary",
    data = True,
    year = "2022",
    com = "13.6",
    lumi = 26.7,
    lumi_format="{0:.1f}",
    ax = ax
)

if not os.path.isdir(args.plot_path):
    os.system(f"mkdir -p {args.plot_path}")

if args.ext != "":
    ext_str = f"_{args.ext}"
else:
    ext_str = ""

plt.savefig(f"{args.plot_path}/zee_{args.detector}_{args.dataset}_weight{ext_str}.pdf")
plt.savefig(f"{args.plot_path}/zee_{args.detector}_{args.dataset}_weight{ext_str}.png")

ax.cla()
