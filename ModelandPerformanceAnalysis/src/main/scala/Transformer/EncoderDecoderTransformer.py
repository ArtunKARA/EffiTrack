from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, MultiHeadAttention, LayerNormalization, Add, Flatten
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# SparkSession başlatma
spark = SparkSession.builder \
    .appName("EncoderDecoderTransformer") \
    .master("local[*]") \
    .getOrCreate()
# Veri setini yükleme
data_path = "C:\\Users\\Artun\\Desktop\\Dosyalar\\github_repos\\EffiTrack\\Data\\HRSS_SMOTE_standard.csv"
raw_data = spark.read.option("header", "true").option("inferSchema", "true").csv(data_path)
# Özellik sütunlarını seçme (Labels hariç)
feature_cols = [col for col in raw_data.columns if col != "Labels"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
assembled_data = assembler.transform(raw_data)
# Veriyi ölçeklendirme
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)
scaler_model = scaler.fit(assembled_data)
scaled_data = scaler_model.transform(assembled_data)
# Özellik vektörlerini ve etiketleri numpy dizilerine dönüştürme
features = np.array(scaled_data.select("scaledFeatures").rdd.map(lambda x: x[0].toArray()).collect())
labels = np.array(raw_data.select("Labels").collect()).flatten()  # Etiketler
# %20 doğrulama için veri ayırma
# validation_split oranı
validation_split = 0.2

# Verileri dengeli bir şekilde eğitim ve doğrulama setlerine ayır
train_features, val_features, train_labels, val_labels = train_test_split(
    features, labels, test_size=validation_split, random_state=42, stratify=labels
)
# Verileri Transformer için yeniden şekillendirme
train_features = train_features.reshape(train_features.shape[0], train_features.shape[1], 1)
val_features = val_features.reshape(val_features.shape[0], val_features.shape[1], 1)
# Etiketlerin boyutunu doğrulama
train_labels = np.squeeze(train_labels)
val_labels = np.squeeze(val_labels)
# Encoder-Decoder Transformer
def encoder_block(inputs, num_heads, key_dim, ff_dim, dropout_rate=0.1):
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(inputs, inputs)
    attention_output = Dropout(dropout_rate)(attention_output)
    attention_output = Add()([inputs, attention_output])
    attention_output = LayerNormalization()(attention_output)
    ff_output = Dense(ff_dim, activation="relu")(attention_output)
    ff_output = Dense(inputs.shape[-1])(ff_output)
    ff_output = Dropout(dropout_rate)(ff_output)
    ff_output = Add()([attention_output, ff_output])
    return LayerNormalization()(ff_output)
def decoder_block(encoder_outputs, decoder_inputs, num_heads, key_dim, ff_dim, dropout_rate=0.1):
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(decoder_inputs, decoder_inputs)
    attention_output = Dropout(dropout_rate)(attention_output)
    attention_output = Add()([decoder_inputs, attention_output])
    attention_output = LayerNormalization()(attention_output)
    cross_attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(
        attention_output, encoder_outputs)
    cross_attention_output = Dropout(dropout_rate)(cross_attention_output)
    cross_attention_output = Add()([attention_output, cross_attention_output])
    cross_attention_output = LayerNormalization()(cross_attention_output)
    ff_output = Dense(ff_dim, activation="relu")(cross_attention_output)
    ff_output = Dense(decoder_inputs.shape[-1])(ff_output)
    ff_output = Dropout(dropout_rate)(ff_output)
    ff_output = Add()([cross_attention_output, ff_output])
    return LayerNormalization()(ff_output)
# Model
encoder_inputs = Input(shape=(train_features.shape[1], train_features.shape[2]))
encoder_outputs = encoder_block(encoder_inputs, num_heads=4, key_dim=64, ff_dim=128)
decoder_inputs = Input(shape=(train_features.shape[1], train_features.shape[2]))
decoder_outputs = decoder_block(encoder_outputs, decoder_inputs, num_heads=4, key_dim=64, ff_dim=128)
decoder_outputs = Dense(1, activation="sigmoid")(decoder_outputs)
output_layer = Flatten()(decoder_outputs)  # Çıkışı düzleştir
transformer_model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=output_layer)
# Model derleme
transformer_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
# Early Stopping tanımlama
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# Model eğitimi
history = transformer_model.fit(
    [train_features, train_features], train_labels,  # Encoder ve decoder için aynı giriş
    epochs=100,
    batch_size=32,
    shuffle=True,
    validation_data=([val_features, val_features], val_labels),
    callbacks=[early_stopping]
)
# Tahminler ve anomali tespiti
val_predictions = transformer_model.predict([val_features, val_features]).flatten()
# Boyutları eşitle
min_length = min(len(val_predictions), len(val_labels))
val_predictions = val_predictions[:min_length]
val_labels = val_labels[:min_length]
# Performans Metriklerini Hesaplama Fonksiyonu
def calculate_metrics(threshold, predictions, true_labels):
    anomalies = (predictions > threshold).astype(int)
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
thresholds = np.arange(0.1, 0.9, 0.1)
results = []
for threshold in thresholds:
    metrics = calculate_metrics(threshold, val_predictions, val_labels)
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