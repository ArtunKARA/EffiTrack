import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Veri setini yükleme
file_path = 'C:/Users/Artun/Desktop/Dosyalar/github_repos/EffiTrack/Data/HRSS_anomalous_optimized.csv'
data = pd.read_csv(file_path)

# Sayısal sütunları seçme
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()

# İlk iki sayısal sütun için scatterplot oluşturma
x_column = numerical_columns[0]  # İlk sütun
y_column = numerical_columns[1]  # İkinci sütun

plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x=x_column, y=y_column, hue="Labels", palette="viridis", alpha=0.7)
plt.title(f"Scatterplot: {x_column} vs {y_column}")
plt.xlabel(x_column)
plt.ylabel(y_column)
plt.legend(title="Labels", loc='upper right')
plt.show()
