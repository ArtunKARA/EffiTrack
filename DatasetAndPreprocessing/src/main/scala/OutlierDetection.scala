import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

object OutlierDetection {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("Outlier Detection")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    // Veri setlerini yükleme
    val optimizedDataPath = "C:\\Users\\Artun\\Desktop\\Dosyalar\\github_repos\\EffiTrack\\Data\\HRSS_normal_optimized.csv"
    val standardDataPath = "C:\\Users\\Artun\\Desktop\\Dosyalar\\github_repos\\EffiTrack\\Data\\HRSS_normal_standard.csv"

    val optimizedDF = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(optimizedDataPath)

    val standardDF = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(standardDataPath)

    // Z-Skoru yöntemi ile uç değerlerin tespiti
    def detectOutliersZScore(df: DataFrame, column: String): DataFrame = {
      val stats = df.select(mean(col(column)), stddev(col(column))).first()
      val meanValue = stats.getDouble(0)
      val stdDevValue = stats.getDouble(1)

      println(s"Z-Score Stats for $column: Mean = $meanValue, StdDev = $stdDevValue")

      df.withColumn("zScore", (col(column) - meanValue) / stdDevValue)
        .withColumn("isOutlier", abs(col("zScore")) > 3)
    }

    // Uç değerlerin kaldırılması
    def removeOutliers(df: DataFrame, column: String): DataFrame = {
      val outlierFreeDF = df.filter(!$"isOutlier")
      outlierFreeDF.drop("isOutlier", "zScore")
    }

    // HRSS_normal_optimized için işlemler
    val optimizedColumns = optimizedDF.columns.filter(c => c != "Timestamp" && c != "Labels")
    var cleanedOptimizedDF = optimizedDF

    optimizedColumns.foreach { column =>
      val outlierDF = detectOutliersZScore(cleanedOptimizedDF, column)
      cleanedOptimizedDF = removeOutliers(outlierDF, column)
    }

    cleanedOptimizedDF.write
      .option("header", "true")
      .csv("C:\\Users\\Artun\\Desktop\\Dosyalar\\github_repos\\EffiTrack\\Data\\Cleaned_HRSS_normal_optimized.csv")

    // HRSS_normal_standard için işlemler
    val standardColumns = standardDF.columns.filter(c => c != "Timestamp" && c != "Labels")
    var cleanedStandardDF = standardDF

    standardColumns.foreach { column =>
      val outlierDF = detectOutliersZScore(cleanedStandardDF, column)
      cleanedStandardDF = removeOutliers(outlierDF, column)
    }

    cleanedStandardDF.write
      .option("header", "true")
      .csv("C:\\Users\\Artun\\Desktop\\Dosyalar\\github_repos\\EffiTrack\\Data\\Cleaned_HRSS_normal_standard.csv")

    spark.stop()
  }
}
