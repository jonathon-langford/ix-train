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

# Function to make confusion matrix with correct weights
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(df, labels, normalize=False, normalizeBy='truth', category_pred='category_pred', dropBkg=False, ext=""):
    
    cm = confusion_matrix(df['category'], df[category_pred], sample_weight=df['weight_lumiScaled'])
    
    if normalize:
        if normalizeBy == 'truth':
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        elif normalizeBy == 'pred':
            cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
        else:  
            raise ValueError("normalizeBy must be either 'truth' or 'pred'")
    
    if dropBkg:
        labels_truth = labels[1:]
        cm = cm[1:, :]
    else:
        labels_truth = labels

    fig, ax = plt.subplots(figsize=(12, 12))
    cax = ax.matshow(cm, cmap='Blues')
    fig.colorbar(cax)
 
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels_truth)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(labels_truth)
    ax.xaxis.tick_bottom()
    
    ax.set_xlabel('Predicted', labelpad=10)
    # Set y-axis label with offset
    ax.set_ylabel('True', labelpad=20)
    
    for i in range(len(labels_truth)):
        for j in range(len(labels)):
            ax.text(j, i, f'{cm[i, j]:.2f}', ha='center', va='center', color='black', fontsize=14)
    
    ext_str = f"_{ext}" if ext != "" else ""
    if normalize:
        # Set vmin and vmax for colorbar
        vmin = 0
        vmax = 1
        cax.set_clim(vmin, vmax)

        if normalizeBy == 'truth':
            fig.savefig(f'plots/confusion_matrix_normalized_by_truth{ext_str}.png')
        elif normalizeBy == 'pred':
            fig.savefig(f'plots/confusion_matrix_normalized_by_pred{ext_str}.png')
    else:
        fig.savefig(f'plots/confusion_matrix{ext_str}.png', bbox_inches='tight')
    ax.cla()

# Function to calculate confidence level intervals for spline crossings
from scipy.interpolate import interp1d
def find_crossings(graph, yval, spline_type="cubic", spline_points=1000, remin=True, return_all_intervals=False):

    # Build spline
    f = interp1d(graph[0],graph[1],kind=spline_type)
    x_spline = np.linspace(graph[0].min(),graph[0].max(),spline_points)
    y_spline = f(x_spline)
    spline = (x_spline,y_spline)

    # Remin
    if remin:
        x,y = graph[0],graph[1]
        if y_spline.min() <= 0:
            y = y-y_spline.min()
            y_spline -= y_spline.min()
            # Add new point to graph
            x = np.append(x, x_spline[np.argmin(y_spline)])
            y = np.append(y, 0.)
            # Re-sort
            i_sort = np.argsort(x)
            x = x[i_sort]
            y = y[i_sort]
            graph = (x,y)

    # Extract bestfit
    bestfit = graph[0][graph[1]==0]

    crossings, intervals = [], []
    current = None

    for i in range(len(graph[0])-1):
        if (graph[1][i]-yval)*(graph[1][i+1]-yval) < 0.:
            # Find crossing as inverse of spline between two x points
            mask = (spline[0]>graph[0][i])&(spline[0]<=graph[0][i+1])
            f_inv = interp1d(spline[1][mask],spline[0][mask])

            # Find crossing point for catch when yval is out of domain of spline points (unlikely)
            if yval > spline[1][mask].max(): cross = f_inv(spline[1][mask].max())
            elif yval <= spline[1][mask].min(): cross = f_inv(spline[1][mask].min())
            else: cross = f_inv(yval)

            # Add information for crossings
            if ((graph[1][i]-yval) > 0.)&( current is None ):
                current = {
                    'lo':cross,
                    'hi':graph[0][-1],
                    'valid_lo': True,
                    'valid_hi': False
                }
            if ((graph[1][i]-yval) < 0.)&( current is None ):
                current = {
                    'lo':graph[0][0],
                    'hi':cross,
                    'valid_lo': False,
                    'valid_hi': True
                }
            if ((graph[1][i]-yval) < 0.)&( current is not None ):
                current['hi'] = cross
                current['valid_hi'] = True
                intervals.append(current)
                current = None

            crossings.append(cross)

    if current is not None:
        intervals.append(current)

    if len(intervals) == 0:
        current = {
            'lo':graph[0][0],
            'hi':graph[0][-1],
            'valid_lo': False,
            'valid_hi': False
        }
        intervals.append(current)

    for interval in intervals:
        interval['contains_bf'] = False
        if (interval['lo']<=bestfit)&(interval['hi']>=bestfit): interval['contains_bf'] = True

    for interval in intervals:
        if interval['contains_bf']:
            val = (bestfit, interval['hi']-bestfit, interval['lo']-bestfit)

    if return_all_intervals:
        return val, intervals
    else:
        return val


# Function to add results to label
def add_res_to_label(val):
    return "$%.2f^{+%.2f}_{-%.2f}$"%(val[0],abs(val[1]),abs(val[2]))

# Option to calculate the Hessian matrix using finite differences
def finite_diff_hessian(f, mu_prof, mu_fixed, counts, profiled_idx, fixed_idx, eps=1e-5):
    n = len(mu_prof)
    H = np.zeros((n, n))
    fx = f(mu_prof, mu_fixed, counts, profiled_idx, fixed_idx)

    for i in range(n):
        for j in range(n):
            x_ijp = mu_prof.copy()
            x_ijm = mu_prof.copy()
            x_ipj = mu_prof.copy()
            x_imj = mu_prof.copy()

            x_ijp[i] += eps
            x_ijp[j] += eps

            x_ijm[i] += eps
            x_ijm[j] -= eps

            x_ipj[i] -= eps
            x_ipj[j] += eps

            x_imj[i] -= eps
            x_imj[j] -= eps

            fx_ijp = f(x_ijp, mu_fixed, counts, profiled_idx, fixed_idx)
            fx_ijm = f(x_ijm, mu_fixed, counts, profiled_idx, fixed_idx)
            fx_ipj = f(x_ipj, mu_fixed, counts, profiled_idx, fixed_idx)
            fx_imj = f(x_imj, mu_fixed, counts, profiled_idx, fixed_idx)
            # Calculate the second derivative using finite differences
            H[i, j] = (fx_ijp - fx_ijm - fx_ipj + fx_imj) / (4 * eps ** 2)

    return H