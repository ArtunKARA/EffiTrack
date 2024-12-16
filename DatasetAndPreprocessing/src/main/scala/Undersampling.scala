import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import scala.util.Try
import java.io.File

object Undersampling {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("Undersampling")
      .master("local[*]") // Adjust as per your cluster setup
      .getOrCreate()

    try {
      // Load the dataset
      val filePath = "C:\\Users\\Artun\\Desktop\\Dosyalar\\github_repos\\EffiTrack\\Data\\HRSS_anomalous_standard.csv"
      val data = spark.read.option("header", "true").option("inferSchema", "true").csv(filePath)

      println("Dataset successfully loaded. Sample rows:")
      data.show(5)

      // Print schema to verify column types
      println("Dataset schema:")
      data.printSchema()

      // Define the class column name
      val classColumn = "Labels"

      // Convert class column to string to avoid ClassCastException
      val dataWithClassAsString = data.withColumn(classColumn, col(classColumn).cast("string"))

      println("Dataset after casting class column to string:")
      dataWithClassAsString.show(5)

      // Count instances per class
      val classCounts = dataWithClassAsString.groupBy(classColumn).count().collect()

      if (classCounts.isEmpty) {
        throw new Exception("Class column is empty or not properly defined.")
      }

      // Identify minority and majority classes
      val majorityClass = classCounts.maxBy(_.getLong(1)).getString(0)
      val minorityClass = classCounts.minBy(_.getLong(1)).getString(0)

      println(s"Majority class: $majorityClass, Minority class: $minorityClass")

      // Filter majority class and perform undersampling
      val majorityDF = dataWithClassAsString.filter(col(classColumn) === majorityClass)
      val minorityDF = dataWithClassAsString.filter(col(classColumn) === minorityClass)

      println(s"Majority class count: ${majorityDF.count()}, Minority class count: ${minorityDF.count()}")

      if (majorityDF.count() == 0 || minorityDF.count() == 0) {
        throw new Exception("One of the class DataFrames is empty. Check data distribution.")
      }

      val samplingFraction = minorityDF.count().toDouble / majorityDF.count()
      if (samplingFraction <= 0 || samplingFraction > 1) {
        throw new Exception("Invalid sampling fraction computed: " + samplingFraction)
      }

      println(s"Sampling fraction: $samplingFraction")

      val majoritySampledDF = majorityDF.sample(withReplacement = false, fraction = samplingFraction)

      println("Majority class after sampling:")
      majoritySampledDF.show(5)

      // Combine minority and sampled majority classes
      val balancedDF = minorityDF.union(majoritySampledDF)

      println("Class distribution after undersampling:")
      balancedDF.groupBy(classColumn).count().show()

      // Define the output file path
      val outputFilePath = "C:\\Users\\Artun\\Desktop\\Dosyalar\\github_repos\\EffiTrack\\Data\\Balanced_HRSS.csv"

      // Save the balanced dataset as a single CSV file
      balancedDF.coalesce(1) // Coalesce into a single partition
        .write
        .option("header", "true")
        .mode("overwrite") // Overwrite if file exists
        .csv(outputFilePath)

      println(s"Balanced dataset successfully saved to $outputFilePath")

    } catch {
      case e: Exception =>
        println("An error occurred: " + e.getMessage)
        e.printStackTrace()
    } finally {
      spark.stop()
    }
  }
}
