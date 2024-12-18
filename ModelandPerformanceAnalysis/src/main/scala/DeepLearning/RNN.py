from pyspark.sql import SparkSession
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt  # Matplotlib ekledik

# 1. Spark Oturumunu Baslatma
spark = SparkSession.builder \
    .appName("Anomaly Detection RNN") \
    .master("local[*]") \
    .getOrCreate()

# 2. Veri Setini Okuma
file_path = "C:\\Users\\Artun\\Desktop\\Dosyalar\\github_repos\\EffiTrack\\Data\\HRSS_undersample_optimized.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)
pandas_df = df.toPandas()  # Spark DataFrame'i Pandas'a donusturme

# 3. Veri On Isleme
features = pandas_df.drop(columns=['Labels'])  # Bagimsiz degiskenler
target = pandas_df['Labels']  # Bagimli degisken

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Train-test bolme
X_train, X_val, y_train, y_val = train_test_split(scaled_features, target, test_size=0.2, random_state=42)
X_train = np.expand_dims(X_train, axis=2)  # RNN giris formati
X_val = np.expand_dims(X_val, axis=2)

# 4. RNN Modeli Olusturma
model = Sequential([
    SimpleRNN(32, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Modeli Egitme
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))

# 5. Performans Metriklerini Hesaplama
val_predictions = model.predict(X_val).flatten()  # Tahmin edilen degerler
val_labels = np.array(y_val)

def calculate_metrics(threshold, predictions, true_labels):
    anomalies = (predictions > threshold).astype(int)
    TP = np.sum((anomalies == 1) & (true_labels == 1))
    TN = np.sum((anomalies == 0) & (true_labels == 0))
    FP = np.sum((anomalies == 1) & (true_labels == 0))
    FN = np.sum((anomalies == 0) & (true_labels == 1))

    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    error_rate = (FP + FN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

    return TP, TN, FP, FN, accuracy, precision, recall, f1_score, error_rate

# Esik Degerlerini Test Etme
thresholds = np.arange(0.1, 0.9, 0.1)
results = []

for threshold in thresholds:
    metrics = calculate_metrics(threshold, val_predictions, val_labels)
    results.append((threshold, *metrics))

# Sonuclari Yazdirma
print("Threshold | TP  | TN  | FP  | FN  | Accuracy | Precision | Recall | F1 Score | Error Rate")
for result in results:
    print(f"{result[0]:.2f}      | {result[1]:3} | {result[2]:3} | {result[3]:3} | {result[4]:3} | "
          f"{result[5]:.4f}  | {result[6]:.4f}   | {result[7]:.4f} | {result[8]:.4f} | {result[9]:.4f}")

# En Iyi Esigi Belirleme (F1 Score'a Gore)
best_threshold = max(results, key=lambda x: x[8])  # F1 Score'a gore en iyisini sec
print(f"\nEn Iyi Esik Degeri: {best_threshold[0]:.2f}")
print(f"TP: {best_threshold[1]}, TN: {best_threshold[2]}, FP: {best_threshold[3]}, FN: {best_threshold[4]}")
print(f"Dogruluk (Accuracy): {best_threshold[5]:.4f}, Kesinlik (Precision): {best_threshold[6]:.4f}")
print(f"Duyarlilik (Recall): {best_threshold[7]:.4f}, F Olcumu (F1 Score): {best_threshold[8]:.4f}")
print(f"Hata Orani (Error Rate): {best_threshold[9]:.4f}")

# Egitim ve Dogrulama Kayiplarini Gorsellestirme
plt.plot(history.history['loss'], label='Egitim Kaybi')
plt.plot(history.history['val_loss'], label='Dogrulama Kaybi')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title("Egitim ve Dogrulama Kayiplari")
plt.show()

# Spark Oturumunu Kapatma
spark.stop()
