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

# Options for running script TODO replace with argparse
do_plotting = False
do_evaluation = True
ext = "onlyxgb_resX"
plot_dir = f"plots/{ext}"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

scan_ranges = {
    'GG2H': (0, 2),
    'VBF': (-0.5, 3),
    'VH': (-0.5, 3),
    'TOP': (-0.5, 3),
}

# Define ext_str
ext_str = f"_{ext}" if ext != "" else ""

# Training options and hyperparameters
train_hp = {
    "seed":0,
    "nodes":[50, 50],
    "lr":1e-3,
    "N_epochs":100,
    "batch_size":100000,
    "temp":0.1,
    "init_weight_std":0.1,
    "use_residualX_layer":True,
    "init_weight_final_std":0.1,
    "use_all_features":True,
    "dummy_set":-4.0,
    "include_xgboost_outputs":True,
    "use_cosine_annealing":False,
    "lr_initial":1e-3,
    "lr_final":1e-4,
    "cosine_epoch_cycle":333,
    "use_temp_scheduler":False,
    "temp_initial":0.1,
    "temp_final":0.01
}
set_seed(train_hp['seed'])

# Load outputs of train.py script
# Load dataframe
df_train = pd.read_parquet("train.parquet")
df_test = pd.read_parquet("test.parquet")
# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    labels = pkl.load(f)
colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#984ea3']
category_color_map = {category: color for category, color in zip(labels, colors)}

# Train a neural network model with original features and XGBoost outputs
# XGBoost outputs are first (len(labels)) columns and are used as base of residual connection
probs = [f'prob_{category}' for category in labels]
orig_features = ['leadPhotonPt', 'leadPhotonEta', 'subleadPhotonPt',
                    'subleadPhotonEta', 'subleadPhotonIDMVA', 'leadPhotonIDMVA',
                    'leadJetEta', 'subleadJetPt', 'dijetMass', 'dijetPt',
                    'dijetMinDRJetPho', 'leadJetDiphoDEta', 'subleadJetDiphoDPhi',
                    'leadMuonEta', 'leadMuonPt' ,'leadElectronEta', 'leadElectronPt',
                    'leadJetBTagScore', 'subleadJetBTagScore', 
                    'diphotonMass'
                ]

# Apply z-score transformation to orig_features
print("Applying z-score transformation to original features...")
means, stds = {}, {}
transformed_features = []
for feature in orig_features:
    feature_name_transformed = f'{feature}_T'
    df_train[feature_name_transformed], means[feature], stds[feature] = zscore_transform(df_train[feature], dummy_set=train_hp['dummy_set'])
    df_test[feature_name_transformed], _, _ = zscore_transform(df_test[feature], means[feature], stds[feature], dummy_set=train_hp['dummy_set'])
    transformed_features.append(feature_name_transformed)

# Make combined list of features
total_features = probs + transformed_features

# Make plots of transformed features
if do_plotting:
    fig, ax = plt.subplots(figsize=(12, 12))
    for feature in transformed_features:

        # Upper and lower bound at -5.5, 4
        lower_bound = -4.5
        upper_bound = 4.5

        # Plot normalised histogram groups by category
        # Separate for test and train
        for category, group in df_train.groupby('category'):
            sumw = 100*group['weight_lumiScaled'].sum()
            category_color = category_color_map[category]
            ax.hist(group[feature], bins=40, range=(lower_bound,upper_bound), 
                    label=f'{category} Train', 
                    weights=group['weight_lumiScaled']/sumw,
                    histtype='step', color=category_color_map[category], linewidth=2)
        for category, group in df_test.groupby('category'):
            sumw = 100*group['weight_lumiScaled'].sum()
            category_color = category_color_map[category]
            ax.hist(group[feature], bins=40, range=(lower_bound,upper_bound),
                    label=f'{category} Test', 
                    weights=group['weight_lumiScaled']/sumw,
                    alpha=0.2, color=category_color_map[category])
        
        ax.set_xlabel(feature)
        ax.set_ylabel('Fraction of events')
        ax.legend(loc='best')
        fig.savefig(f'{plot_dir}/{feature}_normalised_histogram.png')
        ax.cla()


# Build Inference-Aware neural networks
if train_hp['use_all_features']:
    model = NetResidualInferenceAwareAllFeatures(
        input_dim=len(total_features),
        nodes=train_hp['nodes'],
        output_dim=len(labels),
        temp=train_hp['temp'],
        init_weight_std=train_hp['init_weight_std'],
        include_xgboost_outputs=train_hp['include_xgboost_outputs']
    )
    features = total_features
else:
    if train_hp['use_residualX_layer']:
        model = NetResidualXInferenceAware(
            input_dim=len(probs),
            nodes=train_hp['nodes'],
            output_dim=len(labels),
            temp=train_hp['temp'],
            init_weight_std=train_hp['init_weight_std'],
            init_weight_final_std=train_hp['init_weight_final_std']
        )
    else:
        model = NetResidualInferenceAware(
            input_dim=len(probs),
            nodes=train_hp['nodes'],
            output_dim=len(labels),
            temp=train_hp['temp'],
            init_weight_std=train_hp['init_weight_std']
        )
    features = probs

print("Training model...")
res = train_network_ia(
    model,
    df_train,
    df_test,
    labels,
    features,
    'category',
    'weight_lumiScaled',
    train_hp=train_hp,
    temp=train_hp['temp'],
    cosine_anneal=train_hp['use_cosine_annealing'],
    use_temp_scheduler=train_hp['use_temp_scheduler']
)

# Extract results
print("Training completed. Extracting results...")
model = res['model']
loss_train = res['loss_train']
loss_test = res['loss_test']
lr_vals = res['lr_vals']
temp_vals = res['temp_vals']

# Plot loss function curves with lr and tmp on two right axes
fig, ax1 = plt.subplots(figsize=(8, 6))
ax1.plot(loss_train, label='Training Loss', color='blue')
ax1.plot(loss_test, label='Test Loss', color='orange')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend(loc='upper right', fontsize=18)
ax2 = ax1.twinx()
ax2.plot(lr_vals, label='Learning Rate', color='green', linestyle='--')
ax2.set_ylabel('Learning Rate', fontsize=14, color='green')
ax2.tick_params(axis='y', labelcolor='green', labelsize=10)
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 80))  # Offset the third y-axis
ax3.plot(temp_vals, label='Temperature', color='red', linestyle=':')
ax3.set_ylabel('Temperature', fontsize=14, color='red')
ax3.tick_params(axis='y', labelcolor='red', labelsize=10)
fig.savefig(f"{plot_dir}/loss_curves{ext_str}.png", bbox_inches='tight')

# Save everything to pkl file
train_results = {
    'train_hp': train_hp,
    'loss_train': loss_train,
    'loss_test': loss_test,
    'lr_vals': lr_vals,
    'temp_vals': temp_vals,
    'model': model,
    'features': features,
    'labels': labels,
    'means': means,
    'stds': stds,
    'ext': ext
}

if do_evaluation:

    df_test[f'category_pred_{ext}_idx'] = model.get_probmax(torch.Tensor(df_test[features].to_numpy()))
    # Map to category labels
    df_test[f'category_pred_{ext}'] = df_test[f'category_pred_{ext}_idx'].map(lambda idx: labels[idx])

    # Make confusion matrices with priors applied
    plot_confusion_matrix(df_test, labels, normalize=False, category_pred=f'category_pred_{ext}', plot_dir=plot_dir, ext=ext)
    plot_confusion_matrix(df_test, labels, normalize=True, normalizeBy='truth', category_pred=f'category_pred_{ext}', plot_dir=plot_dir, ext=ext)
    plot_confusion_matrix(df_test, labels, normalize=True, normalizeBy='pred', category_pred=f'category_pred_{ext}', plot_dir=plot_dir, ext=ext)
    # Plot confusion matrix normalized by pred label, removing bkg contribution
    mask = df_test['category'] != 'BKG'
    plot_confusion_matrix(df_test[mask], labels, normalize=True, normalizeBy='pred', dropBkg=True, category_pred=f'category_pred_{ext}', plot_dir=plot_dir, ext=f'{ext}_removeBKG')

    # Extract counts in each truth x pred category combination
    counts = confusion_matrix(df_test['category'], df_test[f'category_pred_{ext}'], sample_weight=df_test['weight_lumiScaled'])

    # Initialise signal-strength values at 1 and set scan ranges
    mu = np.ones(len(labels)) 

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
    # Save CL interval values
    cl68_fixed, cl68_prof = {}, {}
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
        cl68_fixed[label] = res_fixed
        text = add_res_to_label(res_fixed)
        axs[1].plot(x[label], q[label], label=f'Fix other $\mu=1$: {text}', color=category_color_map[label], ls='--', linewidth=2)
        res_prof = find_crossings((x[label], q_prof[label]), 1)
        cl68_prof[label] = res_prof
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

        fig.savefig(f'{plot_dir}/2NLL_{label}{ext_str}.png')
        axs[0].cla()
        axs[1].cla()

    # Make summary plot of CL intervals
    best_fit_prof = np.array([cl68_prof[label][0][0] for label in labels[1:]])
    error_hi_prof = abs(np.array([cl68_prof[label][1][0] for label in labels[1:]]))
    error_lo_prof = abs(np.array([cl68_prof[label][2][0] for label in labels[1:]]))
    c = np.linspace(0,len(labels[1:])-1, len(labels[1:]))
    fig, ax = plt.subplots(figsize=(10, 6))
    # SM line
    ax.plot((1,1), (c.min()-0.5,c.max()+0.5), color='#e42536')
    ax.errorbar(best_fit_prof, c,
                xerr=(error_lo_prof, error_hi_prof),
                ls='None',
                label='$68\\% CL',
                color='black',
                capsize=4,
                linewidth=2)
    ax.plot(best_fit_prof, c, 'o', markersize=4, color='white', markeredgecolor='black')
    ax.barh(c,0,tick_label=labels[1:])
    ax.set_ylim(-0.5,len(labels[1:])-0.5)
    ax.set_xlim(-0.5, 2.5)
    ax.set_xlabel('Signal strength, $\mu$', fontsize=24)
    for hline in range(0,len(labels[1:])):
        ax.axhline(y=hline+0.5, color='grey', alpha=0.25)
    # Add text values to plot
    pos = 0.9
    for i, label in enumerate(labels[1:]):
        text_label = "$%.2f^{+%.2f}_{-%.2f}$"%(best_fit_prof[i],error_hi_prof[i],error_lo_prof[i])
        ax.text(pos, data_to_axis(ax, (0,c[i]))[1], text_label, horizontalalignment='center', verticalalignment='center', fontsize=16, transform=ax.transAxes)
    fig.savefig(f'{plot_dir}/CL_intervals{ext_str}.png', bbox_inches='tight')

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
    fig.savefig(f'{plot_dir}/correlation_matrix{ext_str}.png', bbox_inches='tight')

    # Save results
    train_results['cl68_fixed'] = cl68_fixed
    train_results['cl68_prof'] = cl68_prof
    train_results['corr'] = corr
    train_results['cov'] = cov
    train_results['hess'] = hess

# Save to pkl file
if not os.path.exists("results_ia"):
    os.makedirs("results_ia")
with open(f"results_ia/train_results{ext_str}.pkl", "wb") as f:
    pkl.dump(train_results, f)
