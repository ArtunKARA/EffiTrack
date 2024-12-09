import pandas as pd
import matplotlib.pyplot as plt

# Veri setini okuma 
file_path = 'C:/Users/Artun/Desktop/Dosyalar/github_repos/DataFluxAI/Data/HRSS_anomalous_standard.csv'  # Dosyanızın tam yolunu buraya yazın
dataset = pd.read_csv(file_path)

# Labels sütununun dağılımını görselleştirme
plt.figure(figsize=(8, 5))
dataset['Labels'].value_counts().sort_index().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Distribution of Labels', fontsize=14)
plt.xlabel('Label', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
