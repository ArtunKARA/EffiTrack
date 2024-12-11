import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object AnomalyDetection {

  def main(args: Array[String]): Unit = {
    // SparkSession başlat
    val spark = SparkSession.builder()
      .appName("Anomaly Detection with Mexican Hat Wavelet")
      .master("local[*]")
      .getOrCreate()

    // CSV dosyalarının yolları
    val standardDataPath = "C:\\Users\\Artun\\Desktop\\Dosyalar\\github_repos\\EffiTrack\\Data\\HRSS_normal_standard.csv"
    val optimizedDataPath = "C:\\Users\\Artun\\Desktop\\Dosyalar\\github_repos\\EffiTrack\\Data\\HRSS_normal_optimized.csv"

    // Veri yükleme
    val standardDF = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(standardDataPath)

    val optimizedDF = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(optimizedDataPath)

    // Meksika Şapkası Dalgacığına göre eşik değeri hesaplama
    def mexicanHatWavelet(value: Double): Double = {
      val sigma = 1.0 // Gauss fonksiyonunun standart sapması
      (1 / (math.sqrt(3 * sigma) * math.pow(math.Pi, 0.25))) *
        (1 - math.pow(value / sigma, 2)) *
        math.exp(-math.pow(value, 2) / (2 * math.pow(sigma, 2)))
    }

    // Anomali işaretleme UDF
    val calculateAnomalyUdf = udf { (value: Double, label: Int) =>
      if (value != null) {
        val waveletValue = mexicanHatWavelet(value)
        if (waveletValue.abs > 0.6) 1 else label
      } else label
    }

    // Tüm sütunlar için işlem yap (Timestamp hariç)
    val columnsToCheck = standardDF.columns.filterNot(c => c == "Timestamp")

    // Anomali kontrolü ve Labels güncellemesi
    val updatedStandardDF = columnsToCheck.foldLeft(standardDF.withColumn("Labels", lit(0))) { (df, column) =>
      df.withColumn("Labels", calculateAnomalyUdf(col(column).cast("double"), col("Labels")))
    }

    val updatedOptimizedDF = columnsToCheck.foldLeft(optimizedDF.withColumn("Labels", lit(0))) { (df, column) =>
      df.withColumn("Labels", calculateAnomalyUdf(col(column).cast("double"), col("Labels")))
    }

    // Anomalileri bulma ve ekrana yazdırma
    println("Anomalies in Standard Data:")
    updatedStandardDF.filter("Labels = 1").show(false)

    println("Anomalies in Optimized Data:")
    updatedOptimizedDF.filter("Labels = 1").show(false)

    // Sonuçları kaydetme
    updatedStandardDF.write
      .option("header", "true")
      .csv("C:\\Users\\Artun\\Desktop\\Dosyalar\\github_repos\\EffiTrack\\Data\\Annotated_HRSS_normal_standard.csv")

    updatedOptimizedDF.write
      .option("header", "true")
      .csv("C:\\Users\\Artun\\Desktop\\Dosyalar\\github_repos\\EffiTrack\\Data\\Annotated_HRSS_normal_optimized.csv")

    // SparkSession durdur
    spark.stop()
  }
}
