import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import mplhep

mplhep.style.use("CMS")
mplhep.style.use({"savefig.bbox": "tight"})

plot_path = "plots"
if not os.isdir(plot_path):
    os.system(f"mkdir -p {plot_path}")

def ComputeWeights(df, cg=0, ctgre=0, chg=0, weight_var="plot_weight"):
    mu = 1 + df['a_cg']*cg + df['b_cg_cg']*cg*cg + \
             df['a_ctgre']*ctgre + df['b_ctgre_ctgre']*ctgre*ctgre + \
             df['a_chg']*chg + df['b_chg_chg']*chg*chg + \
             df['b_cg_ctgre']*cg*ctgre + df['b_cg_chg']*cg*chg + df['b_chg_ctgre']*chg*ctgre
    return df[weight_var]*mu

# Load dataframe and select events
f_name = "ttH_processed_selected_with_smeft_cut_mupcleq90.parquet"
df = pd.read_parquet(f_name)

# Calculate systematic-varied weights
w = df['plot_weight']
w_cg0p5 = ComputeWeights(df,cg=0.5)
w_cg1 = ComputeWeights(df,cg=1.0)

plot_vars = df.columns
nbins = 20

for var in plot_vars:
    print(f" --> Plotting {var}")
    x = df[var]
    mask = (x==x)
    x = x[mask]

    _, ax = plt.subplots(2, 1, gridspec_kw={"height_ratios": [1.5, 1], 'hspace': 0}, sharex=True)

    lo = np.percentile(x, 0.5)
    hi = np.percentile(x, 99)

    hists = {}

    hists['sm'] = np.histogram(x, nbins, (lo,hi), weights=w[mask], density=True)
    hists['cg0p5'] = np.histogram(x, nbins, (lo,hi), weights=w_cg0p5[mask], density=True)
    hists['cg1'] = np.histogram(x, nbins, (lo,hi), weights=w_cg1[mask], density=True)

    mplhep.histplot(
        hists['sm'],
        ax = ax[0],
        label = f"SM ({len(x)}/{len(df)})",
        histtype = "fill",
        flow = "none",
        alpha = 0.5,
        facecolor = '#9c9ca1'
    )

    mplhep.histplot(
        hists['cg0p5'],
        ax = ax[0],
        label = "$c_{G}$ = 0.5",
        histtype = "step",
        flow = "none",
        edgecolor = '#5790fc'
    )

    mplhep.histplot(
        hists['cg1'],
        ax = ax[0],
        label = "$c_{G}$ = 1.0",
        histtype = "step",
        flow = "none",
        edgecolor = '#e42536'
    )

    # Ratio panel
    mplhep.histplot(
        (hists['cg0p5'][0]/hists['sm'][0],hists['sm'][1]),
        ax = ax[1],
        histtype = "step",
        flow = "none",
        color = '#5790fc'
    )

    mplhep.histplot(
        (hists['cg1'][0]/hists['sm'][0],hists['sm'][1]),
        ax = ax[1],
        histtype = "step",
        flow = "none",
        color = '#e42536'
    )

    ax[0].set_xlim(lo,hi)
    ax[1].set_xlim(lo,hi)
    ax[1].set_ylim(0,3)
    ax[1].axhline(1, color='grey', ls='--')


    ax[0].set_ylabel("Event density")
    ax[1].set_ylabel("EFT / SM", loc='center')
    ax[1].set_xlabel(var)
    ax[0].legend(loc='best', fontsize=20)

    # Add label
    mplhep.cms.label(
        "Preliminary",
        data = True,
        year = "2022",
        com = "13.6",
        lumi = 8,
        lumi_format="{0:.1f}",
        ax = ax[0]
    )

    plt.savefig(f"{plot_path}/{var}.png")
    ax[0].cla()
    ax[1].cla()
