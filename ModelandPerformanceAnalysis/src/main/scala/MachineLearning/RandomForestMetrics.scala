import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler}
import org.apache.spark.sql.DataFrame

object RandomForestMetrics {
  def main(args: Array[String]): Unit = {
    // SparkSession başlatma
    val spark = SparkSession.builder
      .appName("RandomForestMetrics")
      .master("local[*]")
      .getOrCreate()

    // Veri setini yükleme
    val dataPath = "C:\\Users\\Artun\\Desktop\\Dosyalar\\github_repos\\EffiTrack\\Data\\HRSS_anomalous_standard.csv"
    val rawData = spark.read.option("header", "true").option("inferSchema", "true").csv(dataPath)

    // Özellik sütunlarını belirleme
    val featureCols = rawData.columns.drop(2)

    // Özellikleri birleştirme (VectorAssembler kullanımı)
    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")
    val assembledData = assembler.transform(rawData)

    // Veriyi ölçeklendirme
    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithStd(true)
      .setWithMean(true)
    val scalerModel = scaler.fit(assembledData)
    val scaledData = scalerModel.transform(assembledData)

    // Etiket sütununu doğru tipte hazırlama
    val finalData = scaledData.withColumnRenamed("Labels", "label")

    // Eğitim ve test verisi olarak ayırma
    val Array(trainingData, testData) = finalData.randomSplit(Array(0.8, 0.2), seed = 1234L)

    // Rastgele Orman modelini oluşturma ve eğitme
    val randomForest = new RandomForestClassifier()
      .setFeaturesCol("scaledFeatures")
      .setLabelCol("label")
      .setNumTrees(2000) // 2 0000 ağaçtan oluşan bir orman
    val rfModel = randomForest.fit(trainingData)

    // Test verisi üzerinde tahmin yapma
    val predictions = rfModel.transform(testData)

    // TP, TN, FP, FN hesaplama
    val tp = predictions.filter("prediction == 1.0 AND label == 1.0").count()
    val tn = predictions.filter("prediction == 0.0 AND label == 0.0").count()
    val fp = predictions.filter("prediction == 1.0 AND label == 0.0").count()
    val fn = predictions.filter("prediction == 0.0 AND label == 1.0").count()

    // Metrikleri hesaplama
    val accuracy = (tp + tn).toDouble / (tp + tn + fp + fn) // Doğruluk
    val precision = tp.toDouble / (tp + fp) // Kesinlik
    val recall = tp.toDouble / (tp + fn) // Duyarlılık
    val f1Score = 2 * ((precision * recall) / (precision + recall)) // F Ölçümü
    val errorRate = (fp + fn).toDouble / (tp + tn + fp + fn) // Hata Oranı

    // Sonuçları yazdırma
    println(s"True Positive (TP): $tp")
    println(s"True Negative (TN): $tn")
    println(s"False Positive (FP): $fp")
    println(s"False Negative (FN): $fn")
    println(s"Doğruluk (Accuracy): $accuracy")
    println(s"Kesinlik (Precision): $precision")
    println(s"Duyarlılık (Recall): $recall")
    println(s"F Ölçümü (F1 Score): $f1Score")
    println(s"Hata Oranı (Error Rate): $errorRate")

    // SparkSession kapatma
    spark.stop()
  }
}
