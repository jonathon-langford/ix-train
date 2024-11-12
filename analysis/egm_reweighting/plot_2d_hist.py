import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import mplhep
import os

from utils import *

mplhep.style.use("CMS") 
mplhep.style.use({"savefig.bbox": "tight"})

kin_var_list = ['probe_pt', 'probe_eta', 'fixedGridRhoAll']
shape_var_list = ['probe_sieie', 'probe_ecalPFClusterIso', 'probe_trkSumPtHollowConeDR03', 'probe_hcalPFClusterIso', 'probe_pfChargedIsoPFPV', 'probe_phiWidth', 'probe_trkSumPtSolidConeDR04', 'probe_r9', 'probe_pfChargedIsoWorstVtx', 'probe_s4', 'probe_etaWidth', 'probe_mvaID', 'probe_sieip']

percentiles = (0.5,99.5)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--detector', dest='detector', default='EB', help='Part of detector: EB/EEp/EE-')
parser.add_argument('--dataset', dest='dataset', default='test', help='Dataset: test/val/train')
parser.add_argument('--xvar', dest='xvar', default='probe_pt', help='Variable on x-axis')
parser.add_argument('--yvar', dest='yvar', default='probe_eta', help='Variable on y-axis')
parser.add_argument('--nbinsx', dest='nbinsx', default=40, type=int, help='Number of bins in x-axis')
parser.add_argument('--nbinsy', dest='nbinsy', default=40, type=int, help='Number of bins in y-axis')
#parser.add_argument('-w','--weight', dest='weight', default='w_post_S2', help='Weight to use in histogram')
parser.add_argument('--ext', dest='ext', default="", help='Extension for saving')
parser.add_argument('--plot-path', dest='plot_path', default="plots", help='Path to save plots')
args = parser.parse_args()

# Plotting options
nbinsx = int(args.nbinsx)
nbinsy = int(args.nbinsy)

input_mc = pd.read_parquet(f"samples_processed/MC_{args.dataset}_{args.detector}_processed.parquet", engine="pyarrow")
input_data = pd.read_parquet(f"samples_processed/Data_{args.dataset}_{args.detector}_processed.parquet", engine="pyarrow")

#_, axs = plt.subplots(1, 3, figsize=(20,10)) #, gridspec_kw={"width_ratios": [1, 1, 1], 'wspace': 0}, sharey=True)
print(f" --> Plotting: {args.xvar} vs {args.yvar}")

xlo = np.percentile(pd.concat([input_mc[args.xvar],input_data[args.xvar]]), percentiles[0])
xhi = np.percentile(pd.concat([input_mc[args.xvar],input_data[args.xvar]]), percentiles[1])

ylo = np.percentile(pd.concat([input_mc[args.yvar],input_data[args.yvar]]), percentiles[0])
yhi = np.percentile(pd.concat([input_mc[args.yvar],input_data[args.yvar]]), percentiles[1])

hists = {}

fig, axs = plt.subplots(1,3, figsize=(21,7))

hists['data'] = np.histogram2d(input_data[args.xvar],input_data[args.yvar],
    bins = (nbinsx,nbinsy),
    range = ((xlo, xhi), (ylo, yhi)),
)

hists['mc'] = np.histogram2d(input_mc[args.xvar],input_mc[args.yvar],
    bins = (nbinsx,nbinsy),
    range = ((xlo, xhi), (ylo, yhi)),
    weights = input_mc['w_post_S1']
)

hists['mc_rwgt'] = np.histogram2d(input_mc[args.xvar],input_mc[args.yvar],
    bins = (nbinsx,nbinsy),
    range = ((xlo, xhi), (ylo, yhi)),
    weights = input_mc['w_post_S2']
)

vmin, vmax = 0, max(hists['data'][0].max(), hists['mc'][0].max(), hists['mc_rwgt'][0].max())

mplhep.hist2dplot(hists['data'],
    cbar = False,
    cmin = vmin,
    cmax = vmax,
    flow = "none",
    ax = axs[0]
)

mplhep.hist2dplot(hists['mc'],
    cbar = False,
    cmin = vmin,
    cmax = vmax,
    flow = "none",
    ax = axs[1]
)

mplhep.hist2dplot(hists['mc_rwgt'],
    cbar = True,
    cbarextend = True,
    cmin = vmin,
    cmax = vmax,
    flow = "none",
    ax = axs[2]
)

axs[0].set_ylabel(var_name_pretty[args.yvar])
for ax in axs:
    ax.set_xlabel(var_name_pretty[args.xvar])

# Add label
mplhep.cms.label(
    "Preliminary",
    data = True,
    rlabel = "",
    ax = axs[0]
)

mplhep.cms.label(
    exp = "",
    data = True,
    llabel = "",
    year = "2022",
    com = "13.6",
    lumi = 26.7,
    lumi_format="{0:.1f}",
    ax = axs[2]
)

axs[0].text(0.1, 0.9, "Data",
    horizontalalignment='left', verticalalignment='center',
    transform=axs[0].transAxes,
    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.5),
    fontsize=18
)

axs[1].text(0.1, 0.9, "MC",
    horizontalalignment='left', verticalalignment='center',
    transform=axs[1].transAxes,
    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.5),
    fontsize=18
)

axs[2].text(0.1, 0.9, "MC, rwgt-corrected",
    horizontalalignment='left', verticalalignment='center',
    transform=axs[2].transAxes,
    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.5),
    fontsize=18
)




if args.ext != "":
    ext_str = f"_{args.ext}"
else:
    ext_str = ""

plt.savefig(f"{args.plot_path}/zee_2d_{args.detector}_{args.dataset}_{args.xvar}_vs_{args.yvar}{ext_str}.pdf")
plt.savefig(f"{args.plot_path}/zee_2d_{args.detector}_{args.dataset}_{args.xvar}_vs_{args.yvar}{ext_str}.png")
