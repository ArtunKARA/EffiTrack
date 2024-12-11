import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dosya yolları
optimized_data_path = 'C:/Users/Artun/Desktop/Dosyalar/github_repos/EffiTrack/Data/HRSS_anomalous_optimized.csv'
standard_data_path = 'C:/Users/Artun/Desktop/Dosyalar/github_repos/EffiTrack/Data/HRSS_anomalous_standard.csv'

# Veri setlerini oku
try:
    df_standard = pd.read_csv(standard_data_path)
    df_optimized = pd.read_csv(optimized_data_path)
    
    # Eğer Timestamp sütunu datetime formatına dönüştürülmemişse dönüştürelim
    df_standard['Timestamp'] = pd.to_datetime(df_standard['Timestamp'])
    df_optimized['Timestamp'] = pd.to_datetime(df_optimized['Timestamp'])

    # Görselleştirme
    plt.figure(figsize=(14, 6))

    # Standard Dataset
    plt.subplot(1, 2, 1)
    sns.lineplot(x='Timestamp', y='Labels', data=df_standard, alpha=0.7, marker='o', linestyle='-', color='blue')
    plt.title("Anomalies Over Time (Standard Dataset)")
    plt.xlabel("Timestamp")
    plt.ylabel("Labels (0=Normal, 1=Anomaly)")

    # Optimized Dataset
    plt.subplot(1, 2, 2)
    sns.lineplot(x='Timestamp', y='Labels', data=df_optimized, alpha=0.7, marker='o', linestyle='-', color='green')
    plt.title("Anomalies Over Time (Optimized Dataset)")
    plt.xlabel("Timestamp")
    plt.ylabel("Labels (0=Normal, 1=Anomaly)")

    plt.tight_layout()
    plt.show()

except FileNotFoundError as e:
    print(f"Dosya bulunamadı: {e}")
except Exception as e:
    print(f"Bir hata oluştu: {e}")
