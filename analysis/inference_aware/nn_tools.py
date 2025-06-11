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
from torch.optim.lr_scheduler import CosineAnnealingLR
import pickle as pkl
from scipy import optimize
from tqdm import tqdm

# Set seed
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

## Neural network tools
def get_batches(arrays, batch_size=None, randomise=False, include_remainder=True):
    length = len(arrays[0])
    idx = np.arange(length)

    if randomise:
        np.random.shuffle(idx)

    n_full_batches = length // batch_size
    is_remainder = (length % batch_size > 0)

    if is_remainder and include_remainder:
        n_batches = n_full_batches + 1
    else:
        n_batches = n_full_batches

    for i_batch in range(n_batches):
        if i_batch < n_full_batches:
            batch_idx = idx[i_batch*batch_size:(i_batch+1)*batch_size]
        else:
            batch_idx = idx[i_batch*batch_size:]

        arrays_batch = [torch.Tensor(array[batch_idx]) for array in arrays]
        yield arrays_batch


# Function to convert Hessian matrix to tensor
def hess_to_tensor(H):
    hess_elements = []
    for i in range(len(H)):
        for j in range(len(H)):
            hess_elements.append(H[i][j].reshape(1))
    return torch.cat(hess_elements).reshape(len(H),len(H))

# Function to compute negative log-likelihood for Asimov data
def NLL_asimov_torch(mu, counts, n_bkg):
    # Build mu vector including factors of 1 for background processes
    mu_total = torch.cat((torch.ones(n_bkg), mu))
    # Sum over different truth procs, multiplying by mu for expected counts
    obs = torch.sum(counts.T, dim=1)
    exp = torch.sum(torch.multiply(counts.T, mu_total), dim=1) 
    # Sum log-likelihoods over different categories
    poisson_terms = -1 * obs * torch.log(exp) + exp
    # Drop bkg category
    return poisson_terms[1:].sum()

# InferenceAware loss function
def InferenceAwareLoss(model, X, y, w, sumw, labels, n_bkg=1):
    # Extract weighted counts
    y_weighted_transpose = torch.multiply(y.T, w)
    # Reweight to get expected counts over full dataset
    sumw_batch = y_weighted_transpose.sum(axis=1)
    y_weighted = torch.multiply(y_weighted_transpose.T, sumw/sumw_batch)

    # Compute model output
    ypred = model(X)

    # Use torch.matmul to extract counts for each true process in each category
    counts = torch.matmul(y_weighted.T, ypred)

    # Build vector of signal-strengths
    mu_vector = []
    for label in labels[n_bkg:]:
        mu_vector.append(torch.tensor(1.0, requires_grad=True))
    mu_vector = torch.stack(mu_vector)

    # Extract Hessian of NLL w.r.t mu_vector
    hess = hess_to_tensor(torch.func.hessian(NLL_asimov_torch)(mu_vector, counts, n_bkg))

    # Compute inverse of hessian and return sum of variances as loss
    cov = torch.linalg.inv(hess)
    return torch.sum(torch.diag(cov))


class NetResidualInferenceAware(nn.Module):
    def __init__(self, input_dim=5, nodes=[50,50], output_dim=5, temp=0.1, init_weight_std=0.1):
        super(NetResidualInferenceAware, self).__init__()
        self.temperature = temp
        self.init_weight_std = init_weight_std
        # Build network
        n_nodes = [input_dim] + nodes + [output_dim]
        self.layers = nn.ModuleList()
        for i in range(len(n_nodes) - 1):
            self.layers.append(nn.Linear(n_nodes[i], n_nodes[i + 1]))
            #self.layers.append(nn.ReLU())
            self.layers.append(nn.Tanh())
        self.init_zero()

    def init_zero(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                # Initialize weights with small non-zero values to avoid zero gradients
                # and biases with zeros
                nn.init.normal_(layer.weight, mean=0.0, std=self.init_weight_std)
                nn.init.zeros_(layer.bias)

    # Function to print model layer weights and biases
    def print_weights(self):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                print(f"Layer {i}:")
                print(f"Weights: {layer.weight.data}")
                print(f"Biases: {layer.bias.data}")
                print("-" * 20)

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

    # Function to return argmax of the model output
    def get_probmax(self, x):
        out = self.layers[0](x)
        for layer in self.layers[1:]:
            out = layer(out)
        out = out + x  # Residual connection
        return torch.argmax(out, dim=1)

# Class for inference-aware neural network with residual connections
# Use all input features with XGBoost probabilities from CE loss to define base of residual
# First len(labels)=output_dim columns are XGBoost probabilities
class NetResidualInferenceAwareAllFeatures(nn.Module):
    def __init__(self, input_dim=25, nodes=[50,50], output_dim=5, 
                 temp=0.1, 
                 init_weight_std=0.1,
                 include_xgboost_outputs=True
                 ):
        super(NetResidualInferenceAwareAllFeatures, self).__init__()
        self.temperature = temp
        self.init_weight_std = init_weight_std
        self.include_xgboost_outputs = include_xgboost_outputs
        # Define number of input/output nodes
        self.output_dim = output_dim
        self.input_dim = input_dim if self.include_xgboost_outputs else input_dim - output_dim
        # Build network
        n_nodes = [self.input_dim] + nodes + [self.output_dim]
        self.layers = nn.ModuleList()
        for i in range(len(n_nodes) - 1):
            self.layers.append(nn.Linear(n_nodes[i], n_nodes[i + 1]))
            #self.layers.append(nn.ReLU())
            self.layers.append(nn.Tanh())
        self.init_zero()

    def init_zero(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                # Initialize weights with small non-zero values to avoid zero gradients
                # and biases with zeros
                nn.init.normal_(layer.weight, mean=0.0, std=self.init_weight_std)
                nn.init.zeros_(layer.bias)

    # Function to print model layer weights and biases
    def print_weights(self):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                print(f"Layer {i}:")
                print(f"Weights: {layer.weight.data}")
                print(f"Biases: {layer.bias.data}")
                print("-" * 20)

    def forward(self, x):
        if self.include_xgboost_outputs:
            # Use all input features including XGBoost outputs
            out = self.layers[0](x)
        else:
            # Use only features excluding XGBoost outputs
            out = self.layers[0](x[:, self.output_dim:])

        for layer in self.layers[1:]:
            out = layer(out)

        out = out + x[:,:self.output_dim]  # Residual connection to first columns of x
        # Apply temperature scaling
        out = out / self.temperature
        return torch.softmax(out, dim=1)
    
    def set_temperature(self, temp):
        self.temperature = temp

    # Function to return argmax of the model output
    def get_probmax(self, x):
        if self.include_xgboost_outputs:
            # Use all input features including XGBoost outputs
            out = self.layers[0](x)
        else:
            # Use only features excluding XGBoost outputs
            out = self.layers[0](x[:, self.output_dim:])

        # Pass through the rest of the layers
        for layer in self.layers[1:]:
            out = layer(out)

        out = out + x[:,:self.output_dim]  # Residual connection
        return torch.argmax(out, dim=1)
    
# Class for inference-aware neural network with residual connections
# Extra linear layer at the end to allow for more complex transformations
class NetResidualXInferenceAware(nn.Module):
    def __init__(self, input_dim=5, nodes=[50,50], output_dim=5, temp=0.1, init_weight_std=0.1, init_weight_final_std=0.01):
        super(NetResidualXInferenceAware, self).__init__()
        self.temperature = temp
        self.init_weight_std = init_weight_std
        self.init_weight_final_std = init_weight_final_std
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Build network
        n_nodes = [input_dim] + nodes + [output_dim]
        self.layers = nn.ModuleList()
        for i in range(len(n_nodes) - 1):
            self.layers.append(nn.Linear(n_nodes[i], n_nodes[i + 1]))
            #self.layers.append(nn.ReLU())
            self.layers.append(nn.Tanh())
        # Add extra linear layer which is initialised as a residual connection
        self.layers.append(nn.Linear(input_dim+output_dim, output_dim))
        self.init_weights()

    def init_weights(self):
        for layer in self.layers[:-1]:  # Exclude the last layer for now
            if isinstance(layer, nn.Linear):
                # Initialize weights with small non-zero values to avoid zero gradients
                # and biases with zeros
                nn.init.normal_(layer.weight, mean=0.0, std=self.init_weight_std)
                nn.init.zeros_(layer.bias)
        # Initialize the last layer to be a residual connection
        last_layer = self.layers[-1]
        w = torch.tensor(np.concatenate([np.eye(self.input_dim),np.zeros((self.input_dim,self.output_dim))]), dtype=torch.float32)
        noise = np.random.normal(0, self.init_weight_final_std, (self.input_dim+self.output_dim,self.output_dim))
        w += torch.tensor(noise, dtype=torch.float32)
        last_layer.weight.data = w.T
        nn.init.zeros_(last_layer.bias)
        

    # Function to print model layer weights and biases
    def print_weights(self):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                print(f"Layer {i}:")
                print(f"Weights: {layer.weight.data}")
                print(f"Biases: {layer.bias.data}")
                print("-" * 20)

    def forward(self, x):
        out = self.layers[0](x)
        for layer in self.layers[1:-1]:
            out = layer(out)
        # Apply residual connection with last layer
        out = self.layers[-1](torch.cat((x, out), dim=1))  # Concatenate input with output
        # Apply temperature scaling
        out = out / self.temperature
        return torch.softmax(out, dim=1)
    
    def set_temperature(self, temp):
        self.temperature = temp

    # Function to return argmax of the model output
    def get_probmax(self, x):
        out = self.layers[0](x)
        for layer in self.layers[1:-1]:
            out = layer(out)
        # Apply residual connection with last layer
        out = self.layers[-1](torch.cat((x, out), dim=1))  # Concatenate input with output
        return torch.argmax(out, dim=1)

# Function for exponentially decaying temperature scheduler
def temp_scheduler(epoch, initial_temp, final_temp, total_epochs):
    decay_rate = (final_temp / initial_temp) ** (1 / total_epochs)
    return initial_temp * (decay_rate ** epoch)

# Function to train inference-aware network
def train_network_ia(model, df_train, df_test, 
                     labels, features, category, weight_var, 
                     temp=0.1, n_bkg=1, train_hp={}, 
                     cosine_anneal=False, use_temp_scheduler=False):

    # Define lists for tracking lr and temp
    lr_vals = []
    temp_vals = []

    # Set temperature for model training
    if use_temp_scheduler:
        model.set_temperature(train_hp['temp_initial'])
    else:
        model.set_temperature(temp)

    # Set random seed for reproducibility
    set_seed(train_hp['seed'])

    # Build optimizer
    if cosine_anneal:
        optimizer = torch.optim.Adam(model.parameters(), lr=train_hp['lr_initial'])
        scheduler = CosineAnnealingLR(optimizer, T_max=train_hp['cosine_epoch_cycle'], eta_min=train_hp['lr_final'])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=train_hp['lr'])

    # Extract X, y and w from dataframes
    X_train = torch.Tensor(df_train[features].to_numpy())
    X_test = torch.Tensor(df_test[features].to_numpy())
    # Build one-hot encoded y labels from 'category'
    y_train = torch.Tensor(pd.get_dummies(df_train[category]).astype(int).to_numpy())
    y_test = torch.Tensor(pd.get_dummies(df_test[category]).astype(int).to_numpy())
    # Extract weights
    w_train = torch.Tensor(df_train[weight_var].to_numpy())
    w_test = torch.Tensor(df_test[weight_var].to_numpy())

    # Derive sum of weights for each true class
    sumw_train = torch.multiply(y_train.T, w_train).sum(axis=1)
    #sumw_test = torch.multiply(y_test.T, w_test).sum(axis=1)

    # Define inference-aware loss
    ia_loss = lambda m, x, y, w: InferenceAwareLoss(m, x, y, w, sumw_train, labels, n_bkg=n_bkg)

    # Store train and test loss
    loss_train, loss_test = [], []

    # Training loop
    print("Starting inference-aware training...")
    with tqdm(range(train_hp['N_epochs'])) as t:
        for i_epoch in t:
            model.train()

            # Generate batches using get_batches function
            batch_gen = get_batches([X_train, y_train, w_train],
                                    batch_size=train_hp['batch_size'],
                                    randomise=True,
                                    include_remainder=False
            )

            # Iterate over batches
            for X_batch, y_batch, w_batch in batch_gen:
                optimizer.zero_grad()
                loss = ia_loss(model, X_batch, y_batch, w_batch)
                loss.backward()
                optimizer.step()

            # Update learning rate if using cosine annealing
            if cosine_anneal:
                scheduler.step()
            lr_vals.append(scheduler.get_last_lr()[0] if cosine_anneal else train_hp['lr']) 

            # Update temperature
            if use_temp_scheduler:
                model.temperature = temp_scheduler(i_epoch,
                                                   train_hp['temp_initial'],
                                                   train_hp['temp_final'],
                                                   train_hp['N_epochs'])
            temp_vals.append(model.temperature)

            # Evaluate loss on train and test set
            model.eval()
            loss_train.append(ia_loss(model, X_train, y_train, w_train).detach())
            loss_test.append(ia_loss(model, X_test, y_test, w_test).detach())
            t.set_postfix(train_loss=loss_train[-1].item(), test_loss=loss_test[-1].item(), 
                          lr=scheduler.get_last_lr()[0] if cosine_anneal else train_hp['lr'],
                          temp=model.temperature if temp_scheduler else temp)

    print("Finished inference-aware training.")
    model.eval()

    returns = {
        'model': model,
        'loss_train': loss_train,
        'loss_test': loss_test,
        'lr_vals': lr_vals,
        'temp_vals': temp_vals
    }
    
    return returns


# Function to apply z-score transformation to an array
# If mean, std not specified, compute them from the array excluding dummy_val
def zscore_transform(x, mean=None, std=None, dummy_val=-999.0, dummy_set=-4, return_encoded=False):
    mask = x != dummy_val
    if mean:
        x_mean = mean
    else:
        x_mean = np.mean(x[mask])
    if std:
        x_std = std
    else:
        x_std = np.std(x[mask])
    
    # Do transformation
    x_t = mask*((x-x_mean)/x_std) + ~mask*dummy_set
    if return_encoded:
        return x_t, x_mean, x_std, mask
    else:
        return x_t, x_mean, x_std
