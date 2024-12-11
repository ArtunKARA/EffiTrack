import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dosya yolları
optimized_data_path = 'C:/Users/Artun/Desktop/Dosyalar/github_repos/EffiTrack/Data/HRSS_anomalous_optimized.csv'
standard_data_path = 'C:/Users/Artun/Desktop/Dosyalar/github_repos/EffiTrack/Data/HRSS_anomalous_standard.csv'

# Veri setlerini oku
df_standard = pd.read_csv(standard_data_path)
df_optimized = pd.read_csv(optimized_data_path)

# Timestamp sütununu datetime formatına çevir
df_standard['Timestamp'] = pd.to_datetime(df_standard['Timestamp'])
df_optimized['Timestamp'] = pd.to_datetime(df_optimized['Timestamp'])

# "power" geçen sütunları seç
power_columns = [col for col in df_standard.columns if 'power' in col]

# Standard Dataset: Power sütunlarının dağılımları
plt.figure(figsize=(14, 10))
for i, col in enumerate(power_columns, 1):
    plt.subplot(4, 2, i)
    sns.histplot(df_standard[col], kde=True, color='blue', bins=50)
    plt.title(f"Distribution of {col} (Standard Dataset)")
    plt.xlabel(col)
    plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Standard Dataset: Power sütunlarının zaman içerisindeki trendleri (örneklenmiş veri)
sampled_standard = df_standard.sample(frac=0.1, random_state=42)  # Örnekleme (isteğe bağlı)
plt.figure(figsize=(14, 10))
for i, col in enumerate(power_columns, 1):
    plt.subplot(4, 2, i)
    sns.lineplot(x='Timestamp', y=col, data=sampled_standard, alpha=0.7)
    plt.title(f"Time Trend of {col} (Sampled Standard Dataset)")
    plt.xlabel("Timestamp")
    plt.ylabel(col)
plt.tight_layout()
plt.show()

# Optimized Dataset: Power sütunlarının dağılımları
plt.figure(figsize=(14, 10))
for i, col in enumerate(power_columns, 1):
    plt.subplot(4, 2, i)
    sns.histplot(df_optimized[col], kde=True, color='green', bins=50)
    plt.title(f"Distribution of {col} (Optimized Dataset)")
    plt.xlabel(col)
    plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Optimized Dataset: Power sütunlarının zaman içerisindeki trendleri (örneklenmiş veri)
sampled_optimized = df_optimized.sample(frac=0.1, random_state=42)  # Örnekleme (isteğe bağlı)
plt.figure(figsize=(14, 10))
for i, col in enumerate(power_columns, 1):
    plt.subplot(4, 2, i)
    sns.lineplot(x='Timestamp', y=col, data=sampled_optimized, alpha=0.7)
    plt.title(f"Time Trend of {col} (Sampled Optimized Dataset)")
    plt.xlabel("Timestamp")
    plt.ylabel(col)
plt.tight_layout()
plt.show()
