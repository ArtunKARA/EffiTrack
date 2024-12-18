from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# SparkSession başlatma
spark = SparkSession.builder \
    .appName("SimplifiedAutoencoderMetrics") \
    .master("local[*]") \
    .getOrCreate()

# Veri setini yükleme
data_path = "C:\\Users\\Artun\\Desktop\\Dosyalar\\github_repos\\EffiTrack\\Data\\HRSS_undersample_optimized.csv"
raw_data = spark.read.option("header", "true").option("inferSchema", "true").csv(data_path)

# Normal (Label=0) ve Anormal (Label=1) verileri ayırma
normal_data = raw_data.filter(raw_data["Labels"] == 0)
anomaly_data = raw_data.filter(raw_data["Labels"] == 1)

# Özellik sütunlarını seçme
feature_cols = [col for col in raw_data.columns if col != "Labels"]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# Normal veriyi hazırla (yalnızca normal veriyi eğiteceğiz)
assembled_normal = assembler.transform(normal_data)
scaler_normal = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)
scaler_model_normal = scaler_normal.fit(assembled_normal)
scaled_normal = scaler_model_normal.transform(assembled_normal)

normal_features = np.array(scaled_normal.select("scaledFeatures").rdd.map(lambda x: x[0].toArray()).collect())
normal_labels = np.zeros(len(normal_features))  # Normal veriler için etiket 0

# Doğrulama aşamasında normal ve anormal verileri bir araya getirebiliriz.
assembled_all = assembler.transform(raw_data)
# Aynı scaler kullanılması için normal_data üzerine fit edilen scaler_model_normal kullanılır
scaled_all = scaler_model_normal.transform(assembled_all)

all_features = np.array(scaled_all.select("scaledFeatures").rdd.map(lambda x: x[0].toArray()).collect())
all_labels = np.array(scaled_all.select("Labels").collect()).flatten()

# Veriyi eğitim ve doğrulama olarak ayırma
# Burada eğitim seti sadece normal verilerden oluşuyor.
validation_split_ratio = 0.2
split_index = int(len(normal_features) * (1 - validation_split_ratio))

train_features = normal_features[:split_index]
val_features_normal = normal_features[split_index:]
val_labels_normal = np.zeros(len(val_features_normal))

# Ayrıca anormal verileri de doğrulama setine ekleyelim:
assembled_anomaly = assembler.transform(anomaly_data)
scaled_anomaly = scaler_model_normal.transform(assembled_anomaly)
anomaly_features = np.array(scaled_anomaly.select("scaledFeatures").rdd.map(lambda x: x[0].toArray()).collect())
anomaly_labels = np.ones(len(anomaly_features))

# Doğrulama seti hem normal hem de anormal verilerden oluşsun:
val_features = np.concatenate([val_features_normal, anomaly_features], axis=0)
val_labels = np.concatenate([val_labels_normal, anomaly_labels], axis=0)

# Autoencoder Modeli
input_dim = train_features.shape[1]
autoencoder = Sequential([
    Dense(64, activation="relu", input_dim=input_dim),
    Dense(32, activation="relu"),
    Dense(64, activation="relu"),
    Dense(input_dim, activation="linear")
])

autoencoder.compile(optimizer="adam", loss="mse")

# Early Stopping - deneme amaçlı kullanabilir veya kapatabilirsiniz
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Model eğitimi (Sadece normal verilerle)
history = autoencoder.fit(
    train_features, train_features,
    epochs=100,            # Epoch sayısını artırabilirsiniz
    batch_size=32,
    shuffle=True,
    validation_data=(val_features_normal, val_features_normal),
    callbacks=[early_stopping]
)

# Yeniden oluşturma hatalarını hesaplama
train_reconstructed = autoencoder.predict(train_features)
val_reconstructed = autoencoder.predict(val_features)

train_reconstruction_error = np.mean((train_features - train_reconstructed) ** 2, axis=1)
val_reconstruction_error = np.mean((val_features - val_reconstructed) ** 2, axis=1)

# Performans Metriklerini Hesaplama Fonksiyonu
def calculate_metrics(threshold, reconstruction_error, true_labels):
    anomalies = (reconstruction_error > threshold).astype(int)
    TP = np.sum((anomalies == 1) & (true_labels == 1))  # Gerçek pozitif
    TN = np.sum((anomalies == 0) & (true_labels == 0))  # Gerçek negatif
    FP = np.sum((anomalies == 1) & (true_labels == 0))  # Yanlış pozitif
    FN = np.sum((anomalies == 0) & (true_labels == 1))  # Yanlış negatif

    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    error_rate = (FP + FN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

    return TP, TN, FP, FN, accuracy, precision, recall, f1_score, error_rate

# Eşik Değerlerini Test Etme
thresholds = np.percentile(train_reconstruction_error, np.arange(10, 100, 10))
results = []

for threshold in thresholds:
    metrics = calculate_metrics(threshold, val_reconstruction_error, val_labels)
    results.append((threshold, *metrics))

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

# Eğitim ve doğrulama kayıplarını görselleştirme
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title("Eğitim ve Doğrulama Kayıpları")
plt.show()

# SparkSession kapatma
spark.stop()
