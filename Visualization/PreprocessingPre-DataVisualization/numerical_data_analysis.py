import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Veri setini okuma
file_path = 'C:/Users/Artun/Desktop/Dosyalar/github_repos/EffiTrack/Data/HRSS_normal_optimized.csv'  # Dosya yolunu buraya girin
dataset = pd.read_csv(file_path)

# Sayısal sütunların seçimi
numerical_columns = dataset.select_dtypes(include=['float64', 'int64']).columns

# --- Sayısal Sütunların Histogramları ---
plt.figure(figsize=(20, 15))
dataset[numerical_columns].hist(bins=50, figsize=(20, 15), color='lightblue', edgecolor='black')
plt.suptitle("Distribution of Numerical Columns", fontsize=16)
plt.show()

# --- Korelasyon Matrisi ---
correlation_matrix = dataset[numerical_columns].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
plt.title("Correlation Matrix", fontsize=16)
plt.show()

# --- Eksik Değer Analizi ---
missing_values = dataset.isnull().sum()
missing_values = missing_values[missing_values > 0]

if not missing_values.empty:
    # Eksik değerleri görselleştirme
    plt.figure(figsize=(10, 5))
    missing_values.plot(kind='bar', color='salmon', edgecolor='black')
    plt.title("Missing Values in Columns", fontsize=16)
    plt.ylabel("Number of Missing Values")
    plt.xlabel("Columns")
    plt.xticks(rotation=45)
    plt.show()
else:
    print("No missing values found in the dataset.")
