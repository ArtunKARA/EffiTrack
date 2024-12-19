from pyspark.sql import SparkSession
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 1. Spark Oturumunu Başlatma
spark = SparkSession.builder \
    .appName("Anomaly Detection Autoencoder") \
    .master("local[*]") \
    .getOrCreate()

# 2. Veri Setini Okuma
file_path = "C:\\Users\\Artun\\Desktop\\Dosyalar\\github_repos\\EffiTrack\\Data\\HRSS_SMOTE_standard.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)
pandas_df = df.toPandas()  # Spark DataFrame'i Pandas'a dönüştürme

# 3. Veri Ön İşleme
features = pandas_df.drop(columns=['Labels'])  # Bağımsız değişkenler
target = pandas_df['Labels']  # Bağımlı değişken

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Train-test bölme
X_train, X_val, y_train, y_val = train_test_split(scaled_features, target, test_size=0.2, random_state=42)

# 4. Autoencoder Modeli Oluşturma
input_dim = X_train.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(64, activation='relu')(input_layer)
encoded = Dense(32, activation='relu')(encoded)
encoded = Dense(16, activation='relu')(encoded)
decoded = Dense(16, activation='relu')(encoded)
decoded = Dense(32, activation='relu')(decoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(input_layer, decoded)

# Model Derleme
autoencoder.compile(optimizer='adam', loss='mse')

# Modeli Eğitme
history = autoencoder.fit(X_train, X_train, 
                          epochs=25, 
                          batch_size=32, 
                          validation_data=(X_val, X_val))

# 5. Eğitim ve doğrulama kayıplarını görselleştirme
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title("Eğitim ve Doğrulama Kayıpları")
plt.show()

# 6. Yeniden Yapılandırma Hatası Hesaplama
X_val_pred = autoencoder.predict(X_val)
reconstruction_error = np.mean(np.power(X_val - X_val_pred, 2), axis=1)

# 7. Eşik Değerini Belirleme ve Performans Metrikleri
thresholds = np.linspace(0, reconstruction_error.max(), 100)
results = []

for threshold in thresholds:
    anomalies = (reconstruction_error > threshold).astype(int)
    TP = np.sum((anomalies == 1) & (y_val == 1))
    TN = np.sum((anomalies == 0) & (y_val == 0))
    FP = np.sum((anomalies == 1) & (y_val == 0))
    FN = np.sum((anomalies == 0) & (y_val == 1))

    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    error_rate = (FP + FN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

    results.append((threshold, TP, TN, FP, FN, accuracy, precision, recall, f1_score, error_rate))

# Sonuçları Yazdırma
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

# Spark Oturumunu Kapatma
spark.stop()
