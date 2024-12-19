from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Spark Oturumu Başlatma
spark = SparkSession.builder \
    .appName("GRU Evaluation") \
    .master("local[*]") \
    .getOrCreate()

# Veriyi Yükleme
data_path = "C:\\Users\\Artun\\Desktop\\Dosyalar\\github_repos\\EffiTrack\\Data\\HRSS_SMOTE_standard.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Pandas'a Dönüştürme
data_pd = df.toPandas()

# Giriş (X) ve Çıkış (y) Değişkenlerini Ayırma
X = data_pd.drop(columns=['Labels'])  # Hedef sütunu 'label' kabul edildi
y = data_pd['Labels']

# Veriyi Eğitim ve Test Olarak Bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ölçeklendirme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Veriyi GRU İçin Yeniden Şekillendirme
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# GRU Modeli Oluşturma
model = Sequential([
    GRU(64, input_shape=(X_train.shape[1], 1), return_sequences=False),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Modeli Eğitme
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, batch_size=32)

# Tahminleri Al
predictions = (model.predict(X_test) > 0.5).astype(int)

# Değerlendirme Metriklerini Hesaplama
thresholds = np.arange(0.1, 1.0, 0.1)
results = []
for threshold in thresholds:
    y_pred_thresh = (model.predict(X_test) > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_thresh).ravel()
    acc = accuracy_score(y_test, y_pred_thresh)
    prec = precision_score(y_test, y_pred_thresh)
    rec = recall_score(y_test, y_pred_thresh)
    f1 = f1_score(y_test, y_pred_thresh)
    err = 1 - acc
    results.append([threshold, tp, tn, fp, fn, acc, prec, rec, f1, err])

# Sonuçları Yazdırma
print("Threshold | TP  | TN  | FP  | FN  | Accuracy | Precision | Recall | F1 Score | Error Rate")
for result in results:
    print(f"{result[0]:.2f}      | {result[1]:3} | {result[2]:3} | {result[3]:3} | {result[4]:3} | "
          f"{result[5]:.4f}  | {result[6]:.4f}   | {result[7]:.4f} | {result[8]:.4f} | {result[9]:.4f}")

# En İyi Eşiği Belirleme (F1 Score'a Göre)
best_threshold = max(results, key=lambda x: x[8])  # F1 Score'a göre en iyisini seç
print(f"\nEn İyi Eşik Değeri: {best_threshold[0]:.2f}")
print(f"TP: {best_threshold[1]}, TN: {best_threshold[2]}, FP: {best_threshold[3]}, FN: {best_threshold[4]}")
print(f"Doğruluk (Accuracy): {best_threshold[5]:.4f}, Kesinlik (Precision): {best_threshold[6]:.4f}")
print(f"Duyarlılık (Recall): {best_threshold[7]:.4f}, F Ölçümü (F1 Score): {best_threshold[8]:.4f}")
print(f"Hata Oranı (Error Rate): {best_threshold[9]:.4f}")

# Eğitim ve Doğrulama Kayıplarını Görselleştirme
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title("Eğitim ve Doğrulama Kayıpları")
plt.show()

# Spark Oturumunu Sonlandırma
spark.stop()
