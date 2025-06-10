# Import libraries and dependencies
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Use mplhep CMS style
import mplhep
mplhep.style.use("CMS")
import glob
import xgboost as xgb
import pickle as pkl
from scipy import optimize

# Load modules
from eval_tools import *

# Define parameters
skip_prior = True # Set to True to skip prior application
prior_bkg = 10 # Hyperparameter for background, set to 10 as default

# Load outputs of train.py script
# Load test dataframe
df_test = pd.read_parquet("test.parquet")
# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    labels = pkl.load(f)
colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#984ea3']
category_color_map = {category: color for category, color in zip(labels, colors)}

# Make nominal confusion matrices
plot_confusion_matrix(df_test, labels, normalize=False)
plot_confusion_matrix(df_test, labels, normalize=True, normalizeBy='truth')
plot_confusion_matrix(df_test, labels, normalize=True, normalizeBy='pred')
# Plot confusion matrix normalized by pred label, removing bkg contribution
mask = df_test['category'] != 'BKG'
plot_confusion_matrix(df_test[mask], labels, normalize=True, normalizeBy='pred', dropBkg=True, ext='removeBKG')

# Apply prior factors to probabilities before classification
# Extract priors as fraction of sumw over whole dataframe
if skip_prior:
    prior_map = {category: 1.0 for category in labels}
    prior_bkg = "Skip"
else:
    sumw_map = {}
    for category in labels:
        sumw_map[category] = df_test[df_test['category'] == category]['weight_lumiScaled'].sum()

    # Calculate sumw over signal classes
    mask = df_test['category'] != 'BKG'
    sumw_total = df_test[mask]['weight_lumiScaled'].sum()
    prior_map = {category: sumw / sumw_total for category, sumw in sumw_map.items()}
    # Set prior for BKG to param
    prior_map['BKG'] = prior_bkg

# Apply priors to probabilities
probs = [f'prob_{category}' for category in labels]
probs_prior = [f'prob_{category}_prior' for category in labels]
df_test[probs_prior] = df_test[probs].multiply(prior_map.values(), axis=1)

# Extract new category labels
df_test['category_pred_prior_idx'] = np.argmax(df_test[probs]*prior_map.values(), axis=1)
# Map category indices to category names
df_test['category_pred_prior'] = df_test['category_pred_prior_idx'].map(lambda idx: labels[idx])

# Make confusion matrices with priors applied
plot_confusion_matrix(df_test, labels, normalize=False, category_pred='category_pred_prior', ext=f'prior{prior_bkg}')
plot_confusion_matrix(df_test, labels, normalize=True, normalizeBy='truth', category_pred='category_pred_prior', ext=f'prior{prior_bkg}')
plot_confusion_matrix(df_test, labels, normalize=True, normalizeBy='pred', category_pred='category_pred_prior', ext=f'prior{prior_bkg}')
# Plot confusion matrix normalized by pred label, removing bkg contribution
mask = df_test['category'] != 'BKG'
plot_confusion_matrix(df_test[mask], labels, normalize=True, normalizeBy='pred', dropBkg=True, category_pred='category_pred_prior', ext=f'prior{prior_bkg}_removeBKG')

# Extract counts in each truth x pred category combination
from sklearn.metrics import confusion_matrix
counts = confusion_matrix(df_test['category'], df_test['category_pred_prior'], sample_weight=df_test['weight_lumiScaled'])

# Initialise signal strength values at 1 and set scan ranges
mu = np.ones(len(labels)) 
scan_ranges = {
    'GG2H': (0, 2),
    'VBF': (-0.5, 3),
    'VH': (-0.5, 3),
    'TOP': (-0.5, 3),
}

# Prepare for plots
axs = []
fig = plt.figure(figsize=(10,12))
left, width = 0.1, 0.8
bottom, height = 0.1, 0.3
ax = fig.add_axes([left, bottom, width, height])
axs.append(ax)
bottom, height = 0.42, 0.48
ax = fig.add_axes([left, bottom, width, height])
axs.append(ax)

# Calculate NLL, fixing/profiling other params
x, q, q_prof, poi_vals_prof = {}, {}, {}, {}
for label_idx, label in enumerate(labels):
    if label == 'BKG':
        continue
    x[label], q[label] = TwoDeltaNLL_fixed(mu, counts, labels, scan_mu=label_idx, scan_range=scan_ranges[label], scan_points=100)
    _, q_prof[label], poi_vals_prof[label] = TwoDeltaNLL_profiled(mu, counts, labels, scan_mu=label_idx, scan_range=scan_ranges[label], scan_points=100)

    # Plot profiled parameters
    profiled_idx = [i for i in range(1,len(labels)) if i != label_idx]
    for i, pidx in enumerate(profiled_idx):
        y = np.array(poi_vals_prof[label]).T[i]
        axs[0].plot(x[label], y, label=f'Profiled $\mu_{{{labels[pidx]}}}$', color=category_color_map[labels[pidx]], linewidth=2)
    axs[0].set_xlabel('Signal strength, $\mu_{%s}$'%label, fontsize=24)
    axs[0].set_ylabel('$\mu_{j}$', fontsize=24)
    axs[0].set_xlim(scan_ranges[label])
    axs[0].axvline(x=1, color='grey', linestyle='--')
    axs[0].legend(loc='best', fontsize=16)

    # Plot NLL curves
    res_fixed = find_crossings((x[label], q[label]), 1)
    text = add_res_to_label(res_fixed)
    axs[1].plot(x[label], q[label], label=f'Fix other $\mu=1$: {text}', color=category_color_map[label], ls='--', linewidth=2)
    res_prof = find_crossings((x[label], q_prof[label]), 1)
    text = add_res_to_label(res_prof)
    axs[1].plot(x[label], q_prof[label], label=f'Profile other $\mu$: {text}', color=category_color_map[label], linewidth=2)
    axs[1].set_ylabel('$2\Delta NLL$', fontsize=24)
    axs[1].set_xlim(scan_ranges[label])
    axs[1].tick_params(axis='x', labelsize=0)
    axs[1].set_ylim(0, 8)
    axs[1].axhline(y=1, color='black', linestyle='--')
    axs[1].axhline(y=4, color='black', linestyle='--')
    axs[1].axvline(x=1, color='grey', linestyle='--')
    axs[1].legend(loc='best', fontsize=20)

    fig.savefig(f'plots/two_delta_nll_{label}_prior{prior_bkg}.png')
    axs[0].cla()
    axs[1].cla()

# Extract correlation matrix between fit parameters by numerical deriving hessian matrix at minimum
mu_prof = np.ones(len(labels) - 1)
mu_fixed = np.array([1.])
profiled_idx = [i for i in range(1, len(labels))]  # Exclude BKG
fixed_idx = [0]  # BKG is always fixed
hess = finite_diff_hessian(
    NLL_asimov_profiled, 
    mu_prof, mu_fixed, counts, labels, profiled_idx, fixed_idx, 
    eps=1e-3
)
# Calculate covariance matrix from hessian
cov = np.linalg.inv(hess)
# Calculate correlation matrix
corr = np.zeros((len(labels) - 1, len(labels) - 1))
for i in range(len(labels) - 1):
    for j in range(len(labels) - 1):
        corr[i, j] = cov[i, j] / np.sqrt(cov[i, i] * cov[j, j])

# Plot correlation matrix
fig, ax = plt.subplots(figsize=(10, 10))
cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks(np.arange(len(labels) - 1))
ax.set_yticks(np.arange(len(labels) - 1))
ax.set_xticklabels(labels[1:], rotation=45)
ax.set_yticklabels(labels[1:])
ax.xaxis.tick_bottom()
ax.set_xlabel('Signal strength, $\mu_{i}$', labelpad=10)
ax.set_ylabel('Signal strength, $\mu_{j}$', labelpad=20)
for i in range(len(labels) - 1):
    for j in range(len(labels) - 1):
        ax.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center', color='black', fontsize=16)
fig.colorbar(cax)
fig.savefig(f'plots/correlation_matrix_prior{prior_bkg}.png', bbox_inches='tight')