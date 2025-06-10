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
import torch
import torch.nn as nn
import pickle as pkl
from scipy import optimize

# Load modules
from nn_tools import *
from eval_tools import *

# Load outputs of train.py script
# Load dataframe
df_train = pd.read_parquet("train.parquet")
df_test = pd.read_parquet("test.parquet")
# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    labels = pkl.load(f)
colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#984ea3']
category_color_map = {category: color for category, color in zip(labels, colors)}

# Train a neural network model with XGBoost output as inputs
probs = [f'prob_{category}' for category in labels]

# Define training hyperparameters
train_hp = {
    "lr":1e-4,
    "batch_size":100000,
    "N_epochs":1000,
    "seed":0
}
set_seed(train_hp['seed'])

model = NetResidualInferenceAware(
    input_dim=len(probs),
    nodes=[50, 50],
    output_dim=len(labels),
    temp=0.1
)

model, loss_train, loss_test = train_network_ia(
    model,
    df_train,
    df_test,
    labels,
    probs,
    'category',
    'weight_lumiScaled',
    train_hp=train_hp,
    temp=0.1
)

# Plot loss function curves
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(loss_train, label='Training Loss', color='blue')
ax.plot(loss_test, label='Test Loss', color='orange')
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.legend(loc='best')
fig.savefig("plots/twostep_ia_loss_curves.png", bbox_inches='tight')
ax.cla()

# Plot confusion matrix for test dataset
# Get argmax predictions for training and test sets
df_test['category_pred_ia_idx'] = model.get_probmax(torch.Tensor(df_test[probs].to_numpy()))
# Map to category labels
df_test['category_pred_ia'] = df_test['category_pred_ia_idx'].map(lambda idx: labels[idx])

# Make confusion matrices with priors applied
plot_confusion_matrix(df_test, labels, normalize=False, category_pred='category_pred_ia', ext=f'ia')
plot_confusion_matrix(df_test, labels, normalize=True, normalizeBy='truth', category_pred='category_pred_ia', ext=f'ia')
plot_confusion_matrix(df_test, labels, normalize=True, normalizeBy='pred', category_pred='category_pred_ia', ext=f'ia')
# Plot confusion matrix normalized by pred label, removing bkg contribution
mask = df_test['category'] != 'BKG'
plot_confusion_matrix(df_test[mask], labels, normalize=True, normalizeBy='pred', dropBkg=True, category_pred='category_pred_ia', ext=f'ia_removeBKG')

# Extract counts in each truth x pred category combination
from sklearn.metrics import confusion_matrix
counts = confusion_matrix(df_test['category'], df_test['category_pred_ia'], sample_weight=df_test['weight_lumiScaled'])

# Initialise signal-strength values at 1 and set scan ranges
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

    fig.savefig(f'plots/two_delta_nll_{label}_ia.png')
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
fig.savefig(f'plots/correlation_matrix_ia.png', bbox_inches='tight')

exit(0)

# Below is code to extract hessian step-by-step from model output
# This is now coded within the InferenceAwareLoss function

# Extract X and y from dataframes
X_train = torch.Tensor(df_train[probs].to_numpy())
X_test = torch.Tensor(df_test[probs].to_numpy())
# Build one-hot encoded y labels from 'category'
y_train = torch.Tensor(pd.get_dummies(df_train['category']).astype(int).to_numpy())
y_test = torch.Tensor(pd.get_dummies(df_test['category']).astype(int).to_numpy())
# Extract weights
w_train = torch.Tensor(df_train['weight_lumiScaled'].to_numpy())
w_test = torch.Tensor(df_test['weight_lumiScaled'].to_numpy())

# Derive sum of weights for each true class
sumw_train = torch.multiply(y_train.T,w_train).sum(axis=1) 
sumw_test = torch.multiply(y_test.T,w_test).sum(axis=1) 

# TODO: apply batching
#X_train = X_train[:1000]
#y_train = y_train[:1000]
#w_train = w_train[:1000]
#X_test = X_test[:1000]
#y_test = y_test[:1000]
#w_test = w_test[:1000]

# Extract counts using torch.matmul
# First weight each true category by the corresponding weight
# Assume this has now been batched
y_train_weighted_transpose = torch.multiply(y_train.T, w_train)
y_test_weighted_transpose = torch.multiply(y_test.T, w_test)
# Reweight to get same sumw
sumw_batch_train = y_train_weighted_transpose.sum(axis=1)
sumw_batch_test = y_test_weighted_transpose.sum(axis=1)
y_train_weighted = torch.multiply(y_train_weighted_transpose.T,sumw_train/sumw_batch_train)
y_test_weighted = torch.multiply(y_test_weighted_transpose.T,sumw_test/sumw_batch_test) 

# Evaluate model
model.set_temperature(0.0000001)
ypred_train = model(X_train)
ypred_test = model(X_test)
# Compute counts using torch.matmul
counts_train = torch.matmul(y_train_weighted.T, ypred_train)
counts_test = torch.matmul(y_test_weighted.T, ypred_test)

# Build vector of signal strengths
mu_vector = []
for label in labels[1:]:
    mu_vector.append(torch.tensor(1.0, requires_grad=True))
mu_vector = torch.stack(mu_vector)

# Extract hessian of NLL w.r.t to last four elements of mu_vector
# Using torch.func.hessian
hess_train = hess_to_tensor(torch.func.hessian(NLL_asimov_torch)(mu_vector, counts_train, 1))
hess_test = hess_to_tensor(torch.func.hessian(NLL_asimov_torch)(mu_vector, counts_test, 1))

# Compute the inverse of the hessian
cov_train = torch.linalg.inv(hess_train)
cov_test = torch.linalg.inv(hess_test)

# Compute the sum of variances for signal strengths
varsum_train = cov_train.diagonal().sum()
varsum_test = cov_test.diagonal().sum()