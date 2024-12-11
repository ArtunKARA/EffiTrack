# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# File paths
optimized_data_path = 'C:/Users/Artun/Desktop/Dosyalar/github_repos/EffiTrack/Data/HRSS_anomalous_optimized.csv'
standard_data_path = 'C:/Users/Artun/Desktop/Dosyalar/github_repos/EffiTrack/Data/HRSS_anomalous_standard.csv'

# Load datasets
standard_data = pd.read_csv(standard_data_path)
optimized_data = pd.read_csv(optimized_data_path)

# Heatmap of correlations
plt.figure(figsize=(14, 8))
plt.title("Correlation Heatmap - Standard Data")
sns.heatmap(standard_data.corr(), annot=False, cmap='coolwarm', linewidths=0.5)
plt.show()

plt.figure(figsize=(14, 8))
plt.title("Correlation Heatmap - Optimized Data")
sns.heatmap(optimized_data.corr(), annot=False, cmap='coolwarm', linewidths=0.5)
plt.show()