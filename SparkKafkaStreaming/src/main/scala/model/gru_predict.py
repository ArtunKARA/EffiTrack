import sys
import pandas as pd
import numpy as np
import tensorflow as tf

# Komut satırı argümanlarını al
model_path = sys.argv[1]
data_path = sys.argv[2]

# Modeli yükle
model = tf.keras.models.load_model(model_path)

# Veriyi yükle
data = pd.read_csv(data_path)

# Labels sütununu ayır
labels = data['Labels']  # Anomalileri içeren sütun
features = data.drop(columns=['Labels'])  # Labels sütunu hariç tüm sütunlar

# Veriyi yeniden şekillendir
features_values = features.values  # Labels dışındaki sütunlar
if features_values.shape[1] < 19:
    padding = np.zeros((features_values.shape[0], 19 - features_values.shape[1]))
    features_values = np.hstack((features_values, padding))

reshaped_features = features_values[:, :19].reshape(-1, 19, 1)  # İlk 19 sütunu al, (None, 19, 1) şekline getir

# Tahmin yap
predictions = model.predict(reshaped_features)

# Sonuçları birleştir ve kaydet
output = pd.DataFrame(features)  # Özellikleri ekle
output['Labels'] = labels  # Orijinal etiketleri ekle
output['predictions'] = predictions  # Model tahminlerini ekle
output.to_csv("predictions.csv", index=False)
