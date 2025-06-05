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

from nn_tools import *

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
    "batch_size":2000,
    "N_epochs":100,
    "seed":0
}
set_seed(train_hp['seed'])

model = NetResidualInferenceAware(
    input_dim=len(probs),
    nodes=[50, 50],
    output_dim=len(labels),
    temp=0.1
)

X_train = df_train[probs].to_numpy()
X_test = df_test[probs].to_numpy()
# Build one-hot encoded y labels from 'category'
y_train = pd.get_dummies(df_train['category']).astype(int).to_numpy()
y_test = pd.get_dummies(df_test['category']).astype(int).to_numpy()