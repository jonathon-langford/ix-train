import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import mplhep

mplhep.style.use("CMS")
mplhep.style.use({"savefig.bbox": "tight"})

# Cut threshold 
thr = 90

def ComputeWeights(df, cg=0, ctgre=0, chg=0, weight_var="plot_weight"):
    mu = 1 + df['a_cg']*cg + df['b_cg_cg']*cg*cg + \
             df['a_ctgre']*ctgre + df['b_ctgre_ctgre']*ctgre*ctgre + \
             df['a_chg']*chg + df['b_chg_chg']*chg*chg + \
             df['b_cg_ctgre']*cg*ctgre + df['b_cg_chg']*cg*chg + df['b_chg_ctgre']*chg*ctgre
    return df[weight_var]*mu

# Load dataframe and select events
f_name = "ttH_processed_selected_with_smeft.parquet"
df = pd.read_parquet(f_name)

mask = df['mass_sel'] == df['mass_sel']
df = df[mask]

# Calculate systematic-varied weights
w = ComputeWeights(df,cg=0.5)
mu = w/df['plot_weight']

# In bins of pt find out percentile
pt = df['HTXS_Higgs_pt_sel']
quantiles = np.linspace(0,100,21)
mu_percentiles = []
pt_centers = []
for i in range(len(quantiles)-1):
    if i == 0:
        lo = 0
    else:
        lo = np.percentile(pt,quantiles[i])
    hi = np.percentile(pt,quantiles[i+1])

    mask = (pt>=lo)&(pt<hi)
    mu_percentiles.append(np.percentile(mu[mask],thr))
    pt_centers.append(lo+0.5*(hi-lo))
p = np.polyfit(pt_centers[:-1], mu_percentiles[:-1], deg=2)
y = p[0]*pt*pt + p[1]*pt + p[2]

# Apply cut: quadratic cut on mu
mask = mu < y
df = df[mask]
df.to_parquet("ttH_processed_selected_with_smeft_cut_mupcleq90.parquet")
