import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os

# --- CONFIG ---
CACHE_DIR = "cache_spatial"
PATH_X = os.path.join(CACHE_DIR, "x_features.pt")
# We need the original raw synapse counts to check correlation
# (You might need to re-run a snippet of preprocessing to get these raw counts if you didn't save them)
# For now, let's assume you can extract them or have them. 

def check_feature_correlations(feature_matrix, feature_names):
    df = pd.DataFrame(feature_matrix.numpy(), columns=feature_names)
    
    # Calculate Correlation Matrix
    corr = df.corr()
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Matrix (Check for Redundancy)")
    plt.show()
    
    print("--- High Correlation Pairs (> 0.9) ---")
    # Check for redundant features
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > 0.9:
                print(f"{corr.columns[i]} <--> {corr.columns[j]}: {corr.iloc[i, j]:.4f}")

# Define your feature names based on your preprocessing.py
feature_names = [
    "Soma_Vol", "Total_Vol", "Total_Len", 
    "Soma_X", "Soma_Y", "Soma_Z", 
    "Centroid_X", "Centroid_Y", "Centroid_Z"
]

# Load your features
if os.path.exists(PATH_X):
    x = torch.load(PATH_X)
    check_feature_correlations(x, feature_names)
else:
    print("Feature file not found.")