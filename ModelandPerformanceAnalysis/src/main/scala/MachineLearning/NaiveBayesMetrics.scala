import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.classification.NaiveBayesModel
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

object NaiveBayes {
  def main(args: Array[String]): Unit = {
    // SparkSession başlatma
    val spark = SparkSession.builder
      .appName("NaiveBayes")
      .master("local[*]")
      .getOrCreate()

    // Veri setini yükleme
    val dataPath = "C:\\Users\\Artun\\Desktop\\Dosyalar\\github_repos\\EffiTrack\\Data\\HRSS_anomalous_standard.csv"
    val rawData = spark.read.option("header", "true").option("inferSchema", "true").csv(dataPath)

    // Negatif değerleri pozitif yapmak için minimum değeri bulma
    val features = rawData.columns.drop(2) // İlk iki sütun Timestamp ve Labels
    val minValues = features.map(f => rawData.select(f).rdd.map(_.getDouble(0)).min())

    // Özellikleri pozitif hale getirme
    val labeledData: RDD[LabeledPoint] = rawData.rdd.map(row => {
      val label = row.getAs[Int]("Labels").toDouble
      val featureValues = features.zipWithIndex.map { case (col, idx) =>
        row.getAs[Double](col) - minValues(idx)
      }
      LabeledPoint(label, Vectors.dense(featureValues.toArray))
    })

    // Veri setini eğitim ve test olarak ayırma
    val Array(trainingData, testData) = labeledData.randomSplit(Array(0.8, 0.2), seed = 1234L)

    // Naive Bayes modelini eğitme
    val model: NaiveBayesModel = NaiveBayes.train(trainingData, lambda = 1.0)

    // Test veri setinde tahmin yapma
    val predictionAndLabels = testData.map { point =>
      val prediction = model.predict(point.features)
      (prediction, point.label)
    }

    // Performans metriklerini hesaplama
    val TP = predictionAndLabels.filter { case (prediction, label) => prediction == 1.0 && label == 1.0 }.count()
    val TN = predictionAndLabels.filter { case (prediction, label) => prediction == 0.0 && label == 0.0 }.count()
    val FP = predictionAndLabels.filter { case (prediction, label) => prediction == 1.0 && label == 0.0 }.count()
    val FN = predictionAndLabels.filter { case (prediction, label) => prediction == 0.0 && label == 1.0 }.count()

    // Metrikler
    val accuracy = (TP + TN).toDouble / (TP + TN + FP + FN) // Doğruluk
    val precision = TP.toDouble / (TP + FP) // Kesinlik
    val recall = TP.toDouble / (TP + FN) // Duyarlılık
    val f1Score = 2 * ((precision * recall) / (precision + recall)) // F Ölçümü (F1 Score)
    val errorRate = (FP + FN).toDouble / (TP + TN + FP + FN) // Hata Oranı

    // Sonuçları yazdırma
    println(s"True Positive (TP): $TP")
    println(s"True Negative (TN): $TN")
    println(s"False Positive (FP): $FP")
    println(s"False Negative (FN): $FN")
    println(s"Doğruluk (Accuracy): $accuracy")
    println(s"Kesinlik (Precision): $precision")
    println(s"Duyarlılık (Recall): $recall")
    println(s"F Ölçümü (F1 Score): $f1Score")
    println(s"Hata Oranı (Error Rate): $errorRate")

    // SparkSession kapatma
    spark.stop()
  }
}
