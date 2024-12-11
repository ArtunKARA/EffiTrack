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

# Örnekleme (isteğe bağlı)
sampled_standard = df_standard.sample(frac=0.1, random_state=42)  # Standart veri setinden örnekleme
sampled_optimized = df_optimized.sample(frac=0.1, random_state=42)  # Optimize edilmiş veri setinden örnekleme

# Standart Dataset için Violin Plot
plt.figure(figsize=(24, 4 * len(power_columns)))
for idx, col in enumerate(power_columns, 1):
    plt.subplot(3, 2, idx)
    sns.violinplot(
        x='Labels', y=col, data=sampled_standard, inner='point', palette='Blues'
    )
    plt.title(f"{col} Distribution by Labels (Standard Dataset)")
    plt.xlabel("Labels (0=Normal, 1=Anomaly)")
    plt.ylabel(col)
plt.tight_layout()
plt.show()

# Optimize Dataset için Violin Plot
plt.figure(figsize=(24, 4 * len(power_columns)))
for idx, col in enumerate(power_columns, 1):
    plt.subplot(3, 2, idx)
    sns.violinplot(
        x='Labels', y=col, data=sampled_optimized, inner='point', palette='Greens'
    )
    plt.title(f"{col} Distribution by Labels (Optimized Dataset)")
    plt.xlabel("Labels (0=Normal, 1=Anomaly)")
    plt.ylabel(col)
plt.tight_layout()
plt.show()
