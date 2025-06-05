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

# Set seed
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

class NetResidualInferenceAware(nn.Module):
    def __init__(self, input_dim=5, nodes=[50,50], output_dim=5, temp=0.1):
        super(NetResidualInferenceAware, self).__init__()
        self.temperature = temp
        # Build network
        n_nodes = [input_dim] + nodes + [output_dim]
        self.layers = nn.ModuleList()
        for i in range(len(n_nodes) - 1):
            self.layers.append(nn.Linear(n_nodes[i], n_nodes[i + 1]))
            self.layers.append(nn.ReLU())
        self.init_zero()

    def init_zero(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        out = self.layers[0](x)
        for layer in self.layers[1:]:
            out = layer(out)
        out = out + x  # Residual connection
        # Apply temperature scaling
        out = out / self.temperature
        return torch.softmax(out, dim=1)
    
    def set_temperature(self, temp):
        self.temperature = temp
    
# Function to train inference-aware network
# WRITE THIS NEXT