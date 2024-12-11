import pandas as pd
import matplotlib.pyplot as plt

# Veri setlerini yükleme
standard_data_path = 'C:/Users/Artun/Desktop/Dosyalar/github_repos/EffiTrack/Data/HRSS_anomalous_standard.csv'
optimized_data_path = 'C:/Users/Artun/Desktop/Dosyalar/github_repos/EffiTrack/Data/HRSS_anomalous_optimized.csv'

standard_data = pd.read_csv(standard_data_path)
optimized_data = pd.read_csv(optimized_data_path)

# Anomalilerin analizi
# Labels sütununda anomali olarak nitelendirilebilecek değerleri inceleyelim
standard_label_counts = standard_data['Labels'].value_counts()
optimized_label_counts = optimized_data['Labels'].value_counts()

# Veri görselleştirme için bar grafiği
plt.figure(figsize=(10, 5))

# Standard Dataset Labels
plt.subplot(1, 2, 1)
standard_label_counts.plot(kind='bar', color='blue', alpha=0.7)
plt.title('Standard Dataset Labels Distribution')
plt.xlabel('Labels')
plt.ylabel('Frequency')

# Optimized Dataset Labels
plt.subplot(1, 2, 2)
optimized_label_counts.plot(kind='bar', color='orange', alpha=0.7)
plt.title('Optimized Dataset Labels Distribution')
plt.xlabel('Labels')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()