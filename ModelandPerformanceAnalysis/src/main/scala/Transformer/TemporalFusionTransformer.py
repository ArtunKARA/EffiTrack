from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM
from sklearn.model_selection import train_test_split
import tensorflow as tf

# 1. Spark Oturumunu Oluştur
spark = SparkSession.builder \
    .appName("TFT_Model_Performance") \
    .master("local[*]") \
    .getOrCreate()

# 2. Veriyi Yükle
file_path = "C:\\Users\\Artun\\Desktop\\Dosyalar\\github_repos\\EffiTrack\\Data\\HRSS_undersample_optimized.csv"
df_spark = spark.read.csv(file_path, header=True, inferSchema=True)

# Spark DataFrame'i Pandas'a Dönüştür (Model için)
df = df_spark.toPandas()

# 3. Veri Hazırlığı
X = df.drop(columns=['label'])  # 'label' tahmin etmek istediğimiz sütun olsun
y = df['label']

# Eğitim ve test setine ayırma
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Temporal Fusion Transformer Basit Model
input_layer = Input(shape=(X_train.shape[1],))
lstm_layer = LSTM(64, return_sequences=False)(tf.expand_dims(input_layer, axis=-1))
output_layer = Dense(1, activation='sigmoid')(lstm_layer)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Modeli eğitme
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32)

# 5. Tahminler
val_predictions = model.predict(X_val).flatten()
val_labels = y_val.to_numpy()

# 6. Performans Metriklerini Hesaplama
thresholds = np.arange(0.1, 0.9, 0.1)
results = []

# Hesaplama Fonksiyonu
for threshold in thresholds:
    TP, TN, FP, FN, accuracy, precision, recall, f1_score, error_rate = calculate_metrics(
        threshold, val_predictions, val_labels)
    results.append((threshold, TP, TN, FP, FN, accuracy, precision, recall, f1_score, error_rate))

# 7. Sonuçları Yazdırma
print("Threshold | TP  | TN  | FP  | FN  | Accuracy | Precision | Recall | F1 Score | Error Rate")
for result in results:
    print(f"{result[0]:.2f}      | {result[1]:3} | {result[2]:3} | {result[3]:3} | {result[4]:3} | "
          f"{result[5]:.4f}  | {result[6]:.4f}   | {result[7]:.4f} | {result[8]:.4f} | {result[9]:.4f}")

# En İyi Eşiği Belirleme (F1 Score'a Göre)
best_threshold = max(results, key=lambda x: x[8])
print(f"\nEn İyi Eşik Değeri: {best_threshold[0]:.2f}")
print(f"TP: {best_threshold[1]}, TN: {best_threshold[2]}, FP: {best_threshold[3]}, FN: {best_threshold[4]}")
print(f"Doğruluk (Accuracy): {best_threshold[5]:.4f}, Kesinlik (Precision): {best_threshold[6]:.4f}")
print(f"Duyarlılık (Recall): {best_threshold[7]:.4f}, F Ölçümü (F1 Score): {best_threshold[8]:.4f}")
print(f"Hata Oranı (Error Rate): {best_threshold[9]:.4f}")

# 8. Eğitim ve Doğrulama Kayıplarını Görselleştirme
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title("Eğitim ve Doğrulama Kayıpları")
plt.show()

# Spark Oturumunu Sonlandırma
spark.stop()
