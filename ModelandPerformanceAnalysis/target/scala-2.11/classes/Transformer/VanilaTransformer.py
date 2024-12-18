import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# 1. Spark Session Oluşturuluyor
spark = SparkSession.builder \
    .appName("Vanilla Transformer with Spark") \
     .master("local[*]") \
    .getOrCreate()

# 2. Veri Setini Yükleme
file_path = "/mnt/data/HRSS_undersample_optimized.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)

# 3. Pandas'a Dönüştürme ve Veri Ön Hazırlama
pdf = df.toPandas()
pdf.fillna(0, inplace=True)

# Veri ve etiket ayrımı
y = pdf['label']  # Etiketler, label kolonunda olduğu varsayıldı
X = pdf.drop(columns=['label'])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Vanilla Transformer Modeli Tanımlama
def vanilla_transformer(input_dim):
    inputs = Input(shape=(input_dim,))
    x = Dense(128, activation="relu")(inputs)
    x = LayerNormalization()(x)
    x = MultiHeadAttention(num_heads=2, key_dim=128)(x, x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(x)  # Binary classification için sigmoid kullanılır
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])
    return model

model = vanilla_transformer(X_train.shape[1])

# 5. Modeli Eğitme
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32,
    verbose=1
)

# 6. Tahminler
val_predictions = model.predict(X_val)
val_labels = y_val.to_numpy()

# 7. Performans Metriklerini Hesaplama Fonksiyonu
def calculate_metrics(threshold, predictions, true_labels):
    anomalies = (predictions > threshold).astype(int)
    TP = np.sum((anomalies == 1) & (true_labels == 1))
    TN = np.sum((anomalies == 0) & (true_labels == 0))
    FP = np.sum((anomalies == 1) & (true_labels == 0))
    FN = np.sum((anomalies == 0) & (true_labels == 1))

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    error_rate = (FP + FN) / (TP + TN + FP + FN)

    return TP, TN, FP, FN, accuracy, precision, recall, f1_score, error_rate

# 8. Eşik Değerlerini Test Etme
thresholds = np.arange(0.1, 0.9, 0.1)
results = []
for threshold in thresholds:
    metrics = calculate_metrics(threshold, val_predictions, val_labels)
    results.append((threshold, *metrics))

# 9. Sonuçları Yazdırma
print("Threshold | TP  | TN  | FP  | FN  | Accuracy | Precision | Recall | F1 Score | Error Rate")
for result in results:
    print(f"{result[0]:.2f}      | {result[1]:3} | {result[2]:3} | {result[3]:3} | {result[4]:3} | "
          f"{result[5]:.4f}  | {result[6]:.4f}   | {result[7]:.4f} | {result[8]:.4f} | {result[9]:.4f}")

# En İyi Eşik Değeri Belirleme (F1 Score'a Göre)
best_threshold = max(results, key=lambda x: x[8])
print(f"\nEn İyi Eşik Değeri: {best_threshold[0]:.2f}")
print(f"TP: {best_threshold[1]}, TN: {best_threshold[2]}, FP: {best_threshold[3]}, FN: {best_threshold[4]}")
print(f"Doğruluk (Accuracy): {best_threshold[5]:.4f}, Kesinlik (Precision): {best_threshold[6]:.4f}")
print(f"Duyarlılık (Recall): {best_threshold[7]:.4f}, F Ölçümü (F1 Score): {best_threshold[8]:.4f}")
print(f"Hata Oranı (Error Rate): {best_threshold[9]:.4f}")

# 10. Küyüp Grafiği
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title("Eğitim ve Doğrulama Kayıpları")
plt.show()

# Spark Oturumunu Kapatma
spark.stop()
