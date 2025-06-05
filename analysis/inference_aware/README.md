# Inference-aware
This code was essentially all written with Github Copilot... impressive! That includes this README.

First clone the github repo and move to the right directory:
```
git clone git@github.com:jonathon-langford/ix-train.git
cd ix-train/analysis/inference_aware
```

Then copy all of the parquet files to the `samples` directory:
```
cp {path-to-samples}/*.parquet samples/
```

Train the classifier with:
```
python3 train.py
```
The scipt uses a simple XGBoost classifier with 5 target nodes: (BKG, GG2H, TOP, VBF, VH). By default the script will also plot the input feature distributions and the classifier probability distributions. Note, the dummy-valued events in the input feature distribution plots are removed for plotting, and the fraction of events remaining is quoted in the legend. 

This script will output the trained model, labels and the test/train dataframes with probabilities appended for further evaluation.

To evaluate the performance of the classifier:
```
python3 evaluate.py
```
The script will apply prior factors to each probability according to the initial yields (sum of weights) for each signal process. There is a hyperparameter that can be tuned to scale the BKG process prior factor. By default this is set to 10. The priors can be configured in the script with the following options:
```
skip_prior = False # Set to True to skip prior application
prior_bkg = 10 # Hyperparameter for background, set to 10 as default
```

The script will output:
* Confusion matrices: nominal, norm-by-truth, norm-by-predicted and norm-by-predicted with the true BKG events removed. The confusion matrices are produced with and without the priors applied.
* 2NLL curves when both fixing and profiling the other parameters. In the figure, the bottom panel shows the profiled values of the other parameters.
* Correlation matrix showing the correlations between fit parameters.

There are two comments in the `train.py` script which indicate where to:
* Do the feature transformation if working with neural networks
* Swap the XGBoost model for a neural network