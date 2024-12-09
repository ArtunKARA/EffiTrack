import org.apache.spark.sql.SparkSession

object HRSSDataPreprocessing {
  def main(args: Array[String]): Unit = {
    // SparkSession başlatma
    val spark = SparkSession.builder
      .appName("HRSS Data Preprocessing")
      .config("spark.master", "local")
      .getOrCreate()

    // Veri setini yükleme
    val dataset = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("C:\\Users\\Artun\\Desktop\\Dosyalar\\github_repos\\DataFluxAI\\Data/HRSS_anomalous_standard.csv")

    // Veri analizi
    dataset.printSchema()
    dataset.show(5)

    // Eksik veri analizi
    val nullCounts = dataset.columns.map(col => {
      dataset.filter(dataset(col).isNull).count()
    })
    println(s"Null Counts: ${nullCounts.mkString(", ")}")

    // Diğer işlemler burada devam edebilir.
    spark.stop()
  }
}
