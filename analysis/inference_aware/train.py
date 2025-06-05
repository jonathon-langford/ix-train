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

# Options for running script
do_plotting = True
do_training = True

dfs = {}
for filename in glob.glob("samples/*.parquet"):
    if ("Data" in filename)|("QCD" in filename):
        continue    
    df = pd.read_parquet(filename)
    proc = filename.split("/")[-1].split("_processed")[0]
    df['type'] = proc

    # Calculate sum of weights
    sumw = df['weight_lumiScaled'].sum()

    # Apply mask to drop rows with negative weights
    mask = df['weight_lumiScaled'] >= 0
    # Calculate new sum of weights
    sumw_pos = df[mask]['weight_lumiScaled'].sum()

    # Scale new weights to match original sum of weights
    df['weight_lumiScaled'] = df['weight_lumiScaled'] * (sumw / sumw_pos)

    dfs[proc] = df[mask]

# Concatenate all DataFrames into a single DataFrame adding type=proc column
df = pd.concat(dfs.values())
print("Dataframes successfully read in and concatenated.")

# Apply pre-selection
mask = (df['leadPhotonIDMVA'] > 0.2)&(df['subleadPhotonIDMVA'] > 0.2)
df = df[mask]

# Function to map types to categories in dataframe
categories = {
    "DiPhotonJetsBox_M40_80" : "BKG",
    "DiPhotonJetsBox_MGG-80toInf" : "BKG",
    "GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf" : "BKG",
    "GJet_Pt-20toInf_DoubleEMEnriched_MGG-40to80" : "BKG",
    "GJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf" : "BKG",
    "QCD_Pt-30to40_DoubleEMEnriched_MGG-80toInf" : "BKG",
    "QCD_Pt-30toInf_DoubleEMEnriched_MGG-40to80" : "BKG",
    "QCD_Pt-40toInf_DoubleEMEnriched_MGG-80toInf" : "BKG",
    "VBF_M125" : "VBF",
    "VH_M125" : "VH",
    "ggH_M125" : "GG2H",
    "tHW_M125" : "TOP",
    "tHq_M125" : "TOP",
    "ttH_M125" : "TOP"
}
def map_type_to_category(df):
    df['category'] = df['type'].map(categories)
    df['category'] = df['category'].fillna('Other')
    return df
# Apply mapping to the DataFrame
df = map_type_to_category(df)

# Define category colors for plotting
colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#984ea3']
category_color_map = {category: color for category, color in zip(df['category'].unique(), colors)}

# List of input features
feature_names = ['leadPhotonPt', 'leadPhotonEta', 'subleadPhotonPt',
                       'subleadPhotonEta', 'subleadPhotonIDMVA', 'leadPhotonIDMVA',
                       'leadJetEta', 'subleadJetPt', 'dijetMass', 'dijetPt',
                       'dijetMinDRJetPho', 'leadJetDiphoDEta', 'subleadJetDiphoDPhi',
                       'leadMuonEta', 'leadMuonPt' ,'leadElectronEta', 'leadElectronPt',
                       'leadJetBTagScore', 'subleadJetBTagScore', 
                       'diphotonMass']

# Loop over input features and plot normalised versions
if do_plotting:
    # Create directory for plots if it does not exist
    if not os.path.exists('plots'):
        os.makedirs('plots')

    fig, ax = plt.subplots(figsize=(12, 12))
    for column in df.columns:
        if column not in feature_names:
            continue
        print("Plotting: ", column)

        # Only plot features not equal to dummy value of -999.0
        mask = df[column] != -999.0

        # Extract 0.5 and 99.5 percentile in total dataframe
        lower_bound = df[mask][column].quantile(0.005)
        upper_bound = df[mask][column].quantile(0.995)

        # Plot normalised histogram grouped by category
        for category, group in df.groupby('category'):
            # Apply mask to group
            mask = group[column] != -999.0
            # Calculate fraction of group passing mask and add information to legend
            sumw_passing_mask = 100*(group[mask]['weight_lumiScaled'].sum() / group['weight_lumiScaled'].sum())
            ax.hist(group[mask][column], bins=40, range=(lower_bound, upper_bound), 
                    label=f"{category}: {sumw_passing_mask:.1f}%", 
                    weights=group[mask]['weight_lumiScaled']/group[mask]['weight_lumiScaled'].sum(), 
                    histtype='step', color=category_color_map[category], linewidth=2)
        ax.set_xlabel(column)
        ax.set_ylabel('Fraction of events')
        ax.legend(loc='best')
        fig.savefig(f'plots/{column}_normalised_histogram.png')
        ax.cla()

# THIS IS WHERE YOU WOULD DO THE FEATURE TRANSFORMATION FOR A NEURAL NETWORK


# Ensure each class has the same sum of weights for training classifier
# Calculate the sum of weights for each category
category_weights = df.groupby('category')['weight_lumiScaled'].sum()
# Normalize the weights for each category and set average weight for each event = 1
df['weight_train'] = df.apply(lambda row: row['weight_lumiScaled'] / category_weights[row['category']], axis=1)
df['weight_train'] = df['weight_train']/df['weight_train'].mean()

# Prepare features and labels
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
test_size = 0.5
features = df[feature_names]
labels = df['category']
# Encode labels to integers
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
# Split data into training and testing sets with weights
X_train, X_test, y_train, y_test, w_train, w_test, df_train, df_test = train_test_split(
    features, labels_encoded, df['weight_train'], df, test_size=test_size, random_state=42)

# Scale df weights by inverse of test set size to get the correct yields
df_train['weight_lumiScaled'] = df_train['weight_lumiScaled'] / (1-test_size)
df_test['weight_lumiScaled'] = df_test['weight_lumiScaled'] / test_size

# THIS IS WHERE YOU SHOULD SWAP OUT THE XGBOOST CLASSIFIER FOR A NEURAL NETWORK
# Train XGBoost classifier
do_training = True
if do_training:
    # Prepare eval set for monitoring loss
    eval_set = [(X_train, y_train), (X_test, y_test)]   

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=200)
    model.fit(X_train, y_train, sample_weight=w_train, eval_set=eval_set, verbose=True)
    # Save the model
    model.save_model('xgboost_classifier.json')
    # Save the label encoder
    import joblib
    joblib.dump(le, 'label_encoder.pkl')
    print("Model trained and saved successfully.")

    # Extract training and validation loss
    history = model.evals_result()
    train_loss = history['validation_0']['mlogloss']
    val_loss = history['validation_1']['mlogloss']
    # Plot training and validation loss
    do_plotting = True
    if do_plotting:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train_loss, label='Training Loss')
        ax.plot(val_loss, label='Validation Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend(loc='best')
        fig.savefig('plots/training_validation_loss.png')
        ax.cla()

else:
    # Load the model (if needed)
    model = xgb.XGBClassifier()
    model.load_model('xgboost_classifier.json')

# Predict on training and testing sets
y_pred_train, y_pred_test = model.predict(X_train), model.predict(X_test)
y_pred_prob_train, y_pred_prob_test = model.predict_proba(X_train), model.predict_proba(X_test)

# Append predictions and probabilities to the DataFrames
df_train['category_pred'] = le.inverse_transform(y_pred_train)
df_test['category_pred'] = le.inverse_transform(y_pred_test)
df_train['category_pred_idx'] = y_pred_train
df_test['category_pred_idx'] = y_pred_test
for class_index, class_name in enumerate(le.classes_):
    df_train[f'prob_{class_name}'] = y_pred_prob_train[:, class_index]
    df_test[f'prob_{class_name}'] = y_pred_prob_test[:, class_index]

# Save test and train dataframes with predictions
df_train.to_parquet('train.parquet')
df_test.to_parquet('test.parquet')

# Plotting probability distributions
if do_plotting:
    fig, ax = plt.subplots(figsize=(12, 12))
    for class_index, class_name in enumerate(le.classes_):
        column = f'prob_{class_name}'
        print("Plotting: ", column)

        # Extract 0.5 and 99.5 percentile in total dataframe
        lower_bound = 0
        upper_bound = 1

        # Plot normalised histogram grouped by category
        for category, group in df_train.groupby('category'):
            ax.hist(group[column], bins=50, range=(lower_bound, upper_bound), 
                    label=f"{category} (train)", 
                    weights=group['weight_lumiScaled']/group['weight_lumiScaled'].sum(), 
                    histtype='step', color=category_color_map[category], linewidth=2)
        for category, group in df_test.groupby('category'):
            ax.hist(group[column], bins=50, range=(lower_bound, upper_bound), 
                    label=f"{category} (test)", 
                    weights=group['weight_lumiScaled']/group['weight_lumiScaled'].sum(), 
                    alpha=0.2, color=category_color_map[category])
        ax.set_xlabel(column)
        ax.set_ylabel('Fraction of events')
        ax.legend(loc='best')
        fig.savefig(f'plots/{column}_normalised_histogram.png')
        ax.cla()    

    # Make plot of category compositions
    stacked_data, stacked_weights = [], []
    for category_idx, category in enumerate(le.classes_):
        group = df_test[df_test['category'] == category]
        stacked_data.append(group['category_pred_idx'])
        stacked_weights.append(group['weight_lumiScaled']) 
    ax.hist(
        stacked_data, 
        bins=len(le.classes_), 
        range=(0, len(le.classes_)),
        weights=stacked_weights, 
        label=le.classes_, 
        histtype='barstacked', 
        color=colors, 
        alpha=0.5,
        linewidth=2
    )
    ax.set_xlabel('Predicted Category')
    ax.set_ylabel('Weighted Count')
    bin_centers = [i + 0.5 for i in range(len(le.classes_))]
    ax.set_xticks(bin_centers)
    ax.set_xticklabels(le.classes_, rotation=45, ha='center')
    ax.legend(loc='best')  
    ax.set_yscale('log')  
    fig.savefig('plots/category_composition.png')
    ax.cla()

    # Repeat without BKG
    stacked_data, stacked_weights = [], []
    for category_idx, category in enumerate(le.classes_):
        if category == 'BKG':
            continue
        group = df_test[df_test['category'] == category]
        stacked_data.append(group['category_pred_idx'])
        stacked_weights.append(group['weight_lumiScaled']) 
    ax.hist(
        stacked_data, 
        bins=len(le.classes_), 
        range=(0, len(le.classes_)),
        weights=stacked_weights, 
        label=le.classes_[1:],  # Skip BKG
        histtype='barstacked', 
        color=colors[1:],  # Skip BKG color
        alpha=0.5,
        linewidth=2
    )
    ax.set_xlabel('Predicted Category')
    ax.set_ylabel('Weighted Count')
    bin_centers = [i + 0.5 for i in range(len(le.classes_))]
    ax.set_xticks(bin_centers)
    ax.set_xticklabels(le.classes_, rotation=45, ha='center')
    ax.legend(loc='best')  
    ax.set_yscale('log')  
    fig.savefig('plots/category_composition_sigonly.png')