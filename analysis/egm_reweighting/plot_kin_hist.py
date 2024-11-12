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

percentiles = (0.5,99.5)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--detector', dest='detector', default='EB', help='Part of detector: EB/EEp/EE-')
parser.add_argument('--dataset', dest='dataset', default='test', help='Dataset: test/val/train')
parser.add_argument('--nbins', dest='nbins', default=40, type=int, help='Number of bins in plot')
parser.add_argument('--ext', dest='ext', default="", help='Extension for saving')
parser.add_argument('--do-variance-panel', dest='do_variance_panel', default=False, action="store_true", help='Add variance panel')
args = parser.parse_args()

# Plotting options
nbins = int(args.nbins)

input_mc = pd.read_parquet(f"samples_processed/MC_{args.dataset}_{args.detector}_processed.parquet", engine="pyarrow")
input_data = pd.read_parquet(f"samples_processed/Data_{args.dataset}_{args.detector}_processed.parquet", engine="pyarrow")

if args.do_variance_panel:
    _, ax = plt.subplots(3, 1, gridspec_kw={"height_ratios": [1.5, 1, 1], 'hspace': 0}, sharex=True, figsize=(10,14))
else:
    _, ax = plt.subplots(2, 1, gridspec_kw={"height_ratios": [1.5, 1], 'hspace': 0}, sharex=True)

for v in kin_var_list:
    print(f" --> Plotting: {v}")
    v_corr = f"{v}_corr"
    
    lo = np.percentile(pd.concat([input_mc[v],input_data[v]]), percentiles[0])
    hi = np.percentile(pd.concat([input_mc[v],input_data[v]]), percentiles[1])
    
    hists = {}
    
    hists['data'] = np.histogram(input_data[v], nbins, (lo,hi), weights=input_data['weight'])
    hists['data_sumw2'] = np.histogram(input_data[v], nbins, (lo,hi), weights=input_data['weight']**2)

    hists['mc_prekin'] = np.histogram(input_mc[v], nbins, (lo,hi), weights=input_mc['weight_norm'])
    hists['mc_prekin_sumw2'] = np.histogram(input_mc[v], nbins, (lo,hi), weights=input_mc['weight_norm']**2)
    
    hists['mc'] = np.histogram(input_mc[v], nbins, (lo,hi), weights=input_mc['w_post_S1'])
    hists['mc_sumw2'] = np.histogram(input_mc[v], nbins, (lo,hi), weights=input_mc['w_post_S1']**2)
    
    hists['mc_rwgt'] = np.histogram(input_mc[v], nbins, (lo,hi), weights=input_mc['w_post_S2'])
    hists['mc_rwgt_sumw2'] = np.histogram(input_mc[v], nbins, (lo,hi), weights=input_mc['w_post_S2']**2)
    
    # Top panel
    mplhep.histplot(
        hists['data'],
        w2 = hists['data_sumw2'][0],
        w2method = "poisson",
        ax = ax[0],
        label = "Data",
        histtype = "fill",
        flow = "none",
        alpha = 0.25,
        facecolor = '#9c9ca1'
    )

    mplhep.histplot(
        hists['mc_prekin'],
        w2 = hists['mc_prekin_sumw2'][0],
        w2method = "poisson",
        ax = ax[0],
        label = "MC (pre kinematic rwgt)",
        histtype = "errorbar",
        flow = "none",
        color = '#a96b59'
    )
    
    mplhep.histplot(
        hists['mc'],
        w2 = hists['mc_sumw2'][0],
        w2method = "poisson",
        ax = ax[0],
        label = "MC",
        histtype = "errorbar",
        flow = "none",
        color = '#5790fc'
    )
    
    mplhep.histplot(
        hists['mc_rwgt'],
        w2 = hists['mc_rwgt_sumw2'][0],
        w2method = "poisson",
        ax = ax[0],
        label = "MC, rwgt-corrected",
        histtype = "errorbar",
        flow = "none",
        color = '#e42536'
    )
    
    
    bin_centers = (hists['data'][1][:-1]+hists['data'][1][1:])/2
    bin_widths = (hists['data'][1][1:]-hists['data'][1][:-1])/2

    # Add stat uncertainty boxes
    for i in range(len(bin_widths)):
        point = (bin_centers[i]-bin_widths[i], hists['data'][0][i]-hists['data_sumw2'][0][i]**0.5)
        rect = matplotlib.patches.Rectangle(point, 2*bin_widths[i], 2*hists['data_sumw2'][0][i]**0.5, edgecolor='black', facecolor='None', hatch='XX')
        ax[0].add_patch(rect)
    
        point = (bin_centers[i]-bin_widths[i], 1-hists['data_sumw2'][0][i]**0.5/hists['data'][0][i])
        rect = matplotlib.patches.Rectangle(point, 2*bin_widths[i], 2*hists['data_sumw2'][0][i]**0.5/hists['data'][0][i], facecolor='#9c9ca1', alpha=0.25, hatch='XX')
        ax[1].add_patch(rect)

        if args.do_variance_panel:
            point = (bin_centers[i]-bin_widths[i], -1*hists['mc_prekin_sumw2'][0][i]**0.5/hists['mc_prekin'][0][i])
            rect = matplotlib.patches.Rectangle(point, 2*bin_widths[i], 2*hists['mc_prekin_sumw2'][0][i]**0.5/hists['mc_prekin'][0][i], facecolor='#a96b59', alpha=0.1)
            ax[2].add_patch(rect)

            point = (bin_centers[i]-bin_widths[i], -1*hists['mc_sumw2'][0][i]**0.5/hists['mc'][0][i])
            rect = matplotlib.patches.Rectangle(point, 2*bin_widths[i], 2*hists['mc_sumw2'][0][i]**0.5/hists['mc'][0][i], facecolor='#5790fc', alpha=0.1)
            ax[2].add_patch(rect)

            point = (bin_centers[i]-bin_widths[i], -1*hists['mc_rwgt_sumw2'][0][i]**0.5/hists['mc_rwgt'][0][i])
            rect = matplotlib.patches.Rectangle(point, 2*bin_widths[i], 2*hists['mc_rwgt_sumw2'][0][i]**0.5/hists['mc_rwgt'][0][i], facecolor='#e42536', alpha=0.1)
            ax[2].add_patch(rect)

    
    # Bottom panel
    mplhep.histplot(
        (1+hists['data_sumw2'][0]**0.5/hists['data'][0],hists['data'][1]),
        ax = ax[1],
        histtype = "step",
        edges = False,
        flow = "none",
        color = 'black'
    )
    
    mplhep.histplot(
        (1-hists['data_sumw2'][0]**0.5/hists['data'][0],hists['data'][1]),
        ax = ax[1],
        histtype = "step",
        edges = False,
        flow = "none",
        color = 'black'
    )
    
    mplhep.histplot(
        (hists['mc_prekin'][0]/hists['data'][0],hists['mc_prekin'][1]),
        yerr = hists['mc_prekin_sumw2'][0]**0.5/hists['data'][0],
        ax = ax[1],
        histtype = "errorbar",
        flow = "none",
        color = '#a96b59'
    )

    mplhep.histplot(
        (hists['mc'][0]/hists['data'][0],hists['mc'][1]),
        yerr = hists['mc_sumw2'][0]**0.5/hists['data'][0],
        ax = ax[1],
        histtype = "errorbar",
        flow = "none",
        color = '#5790fc'
    )
    
    mplhep.histplot(
        (hists['mc_rwgt'][0]/hists['data'][0],hists['mc_rwgt'][1]),
        yerr = hists['mc_rwgt_sumw2'][0]**0.5/hists['data'][0],
        ax = ax[1],
        histtype = "errorbar",
        flow = "none",
        color = '#e42536'
    )

    # Variance panel
    if args.do_variance_panel:
        mplhep.histplot(
            (-1*hists['mc_prekin_sumw2'][0]**0.5/hists['mc_prekin'][0],hists['data'][1]),
            ax = ax[2],
            histtype = "step",
            edges = False,
            flow = "none",
            color = '#a96b59'
        )
        mplhep.histplot(
            (hists['mc_prekin_sumw2'][0]**0.5/hists['mc_prekin'][0],hists['data'][1]),
            ax = ax[2],
            histtype = "step",
            edges = False,
            flow = "none",
            color = '#a96b59'
        )

        mplhep.histplot(
            (-1*hists['mc_sumw2'][0]**0.5/hists['mc'][0],hists['data'][1]),
            ax = ax[2],
            histtype = "step",
            edges = False,
            flow = "none",
            color = '#5790fc'
        )
        mplhep.histplot(
            (hists['mc_sumw2'][0]**0.5/hists['mc'][0],hists['data'][1]),
            ax = ax[2],
            histtype = "step",
            edges = False,
            flow = "none",
            color = '#5790fc'
        )
        mplhep.histplot(
            (-1*hists['mc_rwgt_sumw2'][0]**0.5/hists['mc_rwgt'][0],hists['data'][1]),
            ax = ax[2],
            histtype = "step",
            edges = False,
            flow = "none",
            color = '#e42536'
        )
        mplhep.histplot(
            (hists['mc_rwgt_sumw2'][0]**0.5/hists['mc_rwgt'][0],hists['data'][1]),
            ax = ax[2],
            histtype = "step",
            edges = False,
            flow = "none",
            color = '#e42536'
        )
    
    
    
    ax[0].set_xlim(lo,hi)
    ax[1].set_xlim(lo,hi)
    ax[1].set_ylim(0.78,1.22)
    ax[1].axhline(1, color='grey', ls='--')
    
    if args.do_variance_panel:
        ax[2].set_xlim(lo,hi)
        ax[2].set_xlabel(var_name_pretty[v])
        ax[2].set_ylabel("$\\pm \\frac{\\sqrt{\\sum{w_i^2}}}{\\sum{w_i}}$", loc='center')
        ax[2].axhline(0, color='grey', ls='--')
    else:
        ax[1].set_xlabel(var_name_pretty[v])

    ax[0].set_ylabel("Events")
    ax[1].set_ylabel("MC / data", loc='center')
    
    ax[0].legend(loc='best', fontsize=20)

    if v == "probe_pt":
        ax[0].set_yscale("log")
    elif v == "probe_eta":
        ax[0].set_ylim(ax[0].get_ylim()[0],ax[0].get_ylim()[1]*1.2)

    for be in (bin_centers-bin_widths):
        ax[1].axvline(be, color='grey', alpha=0.1)
        if args.do_variance_panel:
            ax[2].axvline(be, color='grey', alpha=0.1)
    
    # Add label
    mplhep.cms.label(
        "Preliminary",
        data = True,
        year = "2022",
        com = "13.6",
        lumi = 26.7,
        lumi_format="{0:.1f}",
        ax = ax[0]
    )
    
    if not os.path.isdir("plots_kin"):
        os.system("mkdir -p plots_kin")

    if args.ext != "":
        ext_str = f"_{args.ext}"
    else:
        ext_str = ""

    plt.savefig(f"plots_kin/zee_{args.detector}_{args.dataset}_{v}{ext_str}.pdf")
    plt.savefig(f"plots_kin/zee_{args.detector}_{args.dataset}_{v}{ext_str}.png")
    
    ax[0].cla()
    ax[1].cla()
    if args.do_variance_panel:
        ax[2].cla()
