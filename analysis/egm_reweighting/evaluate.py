import xgboost
import numpy as np
import pandas as pd
import pickle as pkl
import sys
import re

# Hyperparameters
tau1 = 1.
tau2 = 1.
maxW = 100.

# Input model files
f_zscore = sys.argv[2]
f_m1 = sys.argv[3]
f_m2 = sys.argv[4]

# Taken from ICENET: sigmoid(f/1-f) = exp
def rw_transform_with_logits(logits, absMax=30):
    logits = np.clip(logits, -absMax, absMax)
    return np.exp(logits)

# Load data events (replace with .parquet)
# May need to batch load for many events?
print(f" --> Reading {sys.argv[1]}")
events = pd.read_csv(sys.argv[1])

if "probe_pfChargedIso" not in events.columns:
    events['probe_pfChargedIso'] = events['probe_pfChargedIsoPFPV']

print(f" --> Evaluating model 1")
# Load model 1 (3D)
tau1 = 1
model1 = xgboost.Booster()
model1.load_model(f_m1)
# No z-score normalisation for model1
input_features = model1.feature_names
X3D = xgboost.DMatrix(events[input_features])
# Evaluate model
logits1 = model1.predict(X3D)
logits1 /= tau1
events['rwgt_1'] = np.clip(rw_transform_with_logits(logits1), 0., 100.)

print(f" --> Evaluating model 2")
# Load model 2
model2 = xgboost.Booster()
model2.load_model(f_m2)
# Apply z-score normalisation for model 2
with open(f_zscore, "rb") as fpkl:
    zscore = pkl.load(fpkl)
input_features = zscore['ids']
X = (events[input_features].to_numpy()-zscore['X_mu'])/zscore['X_std']
X = xgboost.DMatrix(X, feature_names=input_features)
# Evaluate modoel
logits2 = model2.predict(X)
logits2 /= tau2
events['rwgt_2'] = np.clip(rw_transform_with_logits(logits2), 0., 100.)

f_out = re.sub(".csv", "_eval.csv", sys.argv[1].split("/")[-1])
print(f" --> Saving as: samples_eval/{f_out}")
events.to_csv(f"samples_eval/{f_out}")
