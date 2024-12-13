import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

object KNNMetrics {
  def main(args: Array[String]): Unit = {
    // SparkSession başlatma
    val spark = SparkSession.builder
      .appName("KNNMetrics")
      .master("local[*]")
      .getOrCreate()

    // Veri setini yükleme
    val dataPath = "C:\\Users\\Artun\\Desktop\\Dosyalar\\github_repos\\EffiTrack\\Data\\HRSS_anomalous_standard.csv"
    val rawData = spark.read.option("header", "true").option("inferSchema", "true").csv(dataPath)

    // Özellik sütunlarını seçme (Timestamp ve Labels dışındaki sütunlar)
    val featureCols = rawData.columns.drop(2)

    // Veriyi RDD formatına dönüştürme ve özellik vektörlerini hazırlama
    val dataRDD: RDD[LabeledPoint] = rawData.rdd.map(row => {
      val label = row.getAs[Int]("Labels").toDouble
      val features = featureCols.map(col => row.getAs[Double](col))
      LabeledPoint(label, Vectors.dense(features))
    })

    // Eğitim ve test veri setlerini ayırma
    val Array(trainingData, testData) = dataRDD.randomSplit(Array(0.8, 0.2), seed = 1234L)

    // k değerini belirleme
    val k = 5

    // Test verisi üzerinde tahmin yapma
    val predictionsAndLabels = testData.collect().map { testPoint =>
      val neighbors = trainingData.collect().map { trainPoint =>
          val distance = euclideanDistance(testPoint.features.toArray, trainPoint.features.toArray)
          (distance, trainPoint.label)
        }
        .sortBy(_._1) // Mesafeye göre sıralama
        .take(k) // İlk k komşuyu seçme

      // En sık görülen sınıfı belirleme
      val prediction = neighbors.groupBy(_._2).mapValues(_.size).maxBy(_._2)._1
      (prediction, testPoint.label)
    }

    // Performans metriklerini hesaplama
    val tp = predictionsAndLabels.count { case (pred, label) => pred == 1.0 && label == 1.0 }
    val tn = predictionsAndLabels.count { case (pred, label) => pred == 0.0 && label == 0.0 }
    val fp = predictionsAndLabels.count { case (pred, label) => pred == 1.0 && label == 0.0 }
    val fn = predictionsAndLabels.count { case (pred, label) => pred == 0.0 && label == 1.0 }

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

  // Öklid Mesafesi Hesaplama
  def euclideanDistance(vec1: Array[Double], vec2: Array[Double]): Double = {
    Math.sqrt(vec1.zip(vec2).map { case (x, y) => Math.pow(x - y, 2) }.sum)
  }
}
