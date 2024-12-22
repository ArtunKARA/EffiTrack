import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.sys.process._

object KafkaAnomalyDetection {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("KafkaAnomalyDetection")
      .master("local[*]")
      .getOrCreate()

    val pythonExecutable = "C:\\Users\\Artun\\anaconda3\\envs\\pyspark_env\\python.exe"
    val pythonScript = "C:\\Users\\Artun\\Desktop\\Dosyalar\\github_repos\\EffiTrack\\SparkKafkaStreaming\\src\\main\\scala\\model\\gru_predict.py"
    val modelPath = "C:\\Users\\Artun\\Desktop\\Dosyalar\\github_repos\\EffiTrack\\SparkKafkaStreaming\\src\\main\\scala\\model\\gru_model.h5"
    val dataPath = "C:\\Users\\Artun\\Desktop\\Dosyalar\\github_repos\\EffiTrack\\SparkKafkaStreaming\\src\\main\\scala\\data\\HRSS_SMOTE_standard.csv"

    // Python script çağrısı
    val command = Seq(pythonExecutable, pythonScript, modelPath, dataPath)
    val result = command.!

    if (result != 0) {
      println(s"Python script execution failed with code $result")
      sys.exit(1)
    } else {
      println("Python script executed successfully.")
    }

    // Tahmin edilen veriyi yükleme
    val predictionsPath = "predictions.csv"
    val predictions = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(predictionsPath)

    predictions.show()

    // Veriyi işlemeye başla
    val anomalies = predictions.filter("predictions > 0.8").withColumn("topic", lit("anomalies"))
    val normalData = predictions.filter("predictions <= 0.2").withColumn("topic", lit("normal_data"))

    // Anomali oranı
    val totalCount = predictions.count()
    val anomalyCount = anomalies.count()
    val anomalyRatio = anomalyCount.toDouble / totalCount

    println(s"Toplam Veri Sayısı: $totalCount")
    println(s"Anomali Sayısı: $anomalyCount")
    println(f"Anomali Oranı: $anomalyRatio%.4f")

    // Kafka'ya gönderme
    def sendToKafka(df: DataFrame, topic: String): Unit = {
      val jsonDF = df.toJSON

      jsonDF.write
        .format("kafka")
        .option("kafka.bootstrap.servers", "localhost:9092")
        .option("topic", topic)
        .save()
    }

    sendToKafka(anomalies, "anomalies")
    sendToKafka(normalData, "normal_data")

    println("Veriler Kafka'ya başarıyla gönderildi.")
    spark.stop()
  }
}
