import pandas as pd
import matplotlib.pyplot as plt

# Veri setlerini yükleme
standard_data_path = 'C:/Users/Artun/Desktop/Dosyalar/github_repos/EffiTrack/Data/HRSS_anomalous_standard.csv'
optimized_data_path = 'C:/Users/Artun/Desktop/Dosyalar/github_repos/EffiTrack/Data/HRSS_anomalous_optimized.csv'

standard_data = pd.read_csv(standard_data_path)
optimized_data = pd.read_csv(optimized_data_path)

# Histogramları çizmek için sütun seçimi
columns_to_plot = ['O_w_BLO_power', 'O_w_BHL_power', 'O_w_BHR_power', 'O_w_BRU_power',
                   'O_w_HR_power', 'O_w_HL_power']

# Histogramları çizme
plt.figure(figsize=(12, 8))
for i, column in enumerate(columns_to_plot, start=1):
    plt.subplot(2, 3, i)
    standard_data[column].hist(bins=50, alpha=0.7, label='Standard', color='blue')
    optimized_data[column].hist(bins=50, alpha=0.7, label='Optimized', color='orange')
    plt.title(column)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()

plt.tight_layout()
plt.show()