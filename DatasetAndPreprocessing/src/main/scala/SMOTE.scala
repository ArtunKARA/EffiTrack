import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{Vectors, Vector => MLVector}
import scala.util.Random

object SMOTE {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("SMOTE")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    val filePath = "C:\\Users\\Artun\\Desktop\\Dosyalar\\github_repos\\EffiTrack\\Data\\HRSS_anomalous_standard.csv"
    val df = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(filePath)

    val featureCols = df.columns.filterNot(c => c == "Labels")

    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")

    val data = assembler.transform(df)

    val labelCounts = data.groupBy("Labels").count().collect()
    val minorityClass = labelCounts.minBy(_.getAs[Long]("count")).getAs[Int]("Labels")
    val majorityClass = labelCounts.maxBy(_.getAs[Long]("count")).getAs[Int]("Labels")

    val minorityDF = data.filter($"Labels" === minorityClass)
    val majorityDF = data.filter($"Labels" === majorityClass)

    val minorityVectors = minorityDF.select("features").rdd.map(r => r.getAs[MLVector](0)).collect()

    val k = 5
    val majorityCount = labelCounts.maxBy(_.getAs[Long]("count")).getAs[Long]("count")
    val minorityCount = labelCounts.minBy(_.getAs[Long]("count")).getAs[Long]("count")
    val numToGenerate = (majorityCount - minorityCount).toInt

    println(s"NumToGenerate: $numToGenerate")
    println(s"MinorityVectors Length: ${minorityVectors.length}")

    if (numToGenerate > 0 && minorityVectors.nonEmpty) {
      val syntheticSamples = (1 to numToGenerate).par.map { _ =>
        val original = minorityVectors(Random.nextInt(minorityVectors.length))
        val neighbors = findKNearestNeighbors(original, minorityVectors, k)
        val synth = generateSyntheticSample(original, neighbors).toArray
        (null.asInstanceOf[String], minorityClass, synth)
      }.toList

      val synthDF = syntheticSamples.toDF("Timestamp", "Labels", "featuresArray")

      // Array'ı tekrar Vector'e çevir
      val toVector = udf((arr: Seq[Double]) => Vectors.dense(arr.toArray))
      var synthDFwithVector = synthDF.withColumn("features", toVector($"featuresArray")).drop("featuresArray")

      // Şimdi 'features' kolonundan tekrar orijinal kolonlara dönüştürelim
      val vectorToArrayUDF = udf((v: MLVector) => v.toArray)
      synthDFwithVector = synthDFwithVector.withColumn("featuresArray", vectorToArrayUDF($"features"))

      // Her bir feature kolonu için featuresArray'den ilgili indexteki değeri çek
      val indexedCols = featureCols.zipWithIndex
      indexedCols.foreach { case (colName, idx) =>
        synthDFwithVector = synthDFwithVector.withColumn(colName, $"featuresArray"(idx))
      }

      // Artık features ve featuresArray kolonlarına ihtiyacımız yok
      synthDFwithVector = synthDFwithVector.drop("features").drop("featuresArray")

      // Orijinal dataframede olmayan kolonları null olarak eklemeye gerek yok,
      // çünkü artık tüm orijinal kolonlar geri kazandırıldı.
      // Eğer df üzerinde Timestamp ve Labels harici kolonlar varsa, onları da features'tan türettik.
      // Timestamp null olarak kalır, istenirse başka değer verilebilir.

      // Sütun sırasını korumak için:
      val finalSynthDF = synthDFwithVector.select(df.columns.map(col): _*)

      val balancedDF = majorityDF.select(df.columns.map(col): _*)
        .union(minorityDF.select(df.columns.map(col): _*))
        .union(finalSynthDF.select(df.columns.map(col): _*))

      balancedDF.show(5)
      balancedDF.coalesce(1).write.option("header", "true")
        .csv("C:\\Users\\Artun\\Desktop\\Dosyalar\\github_repos\\EffiTrack\\Data\\Balanced_HRSS_SMOTE.csv")

      println("Dengelenmiş veri seti başarıyla kaydedildi.")
    } else {
      println("Sentetik veri üretilemedi. NumToGenerate sıfır veya MinorityVectors boş.")
    }

    spark.stop()
  }

  def euclideanDist(v1: MLVector, v2: MLVector): Double = {
    val arr1 = v1.toArray
    val arr2 = v2.toArray
    math.sqrt(arr1.zip(arr2).map { case (a, b) => math.pow(a - b, 2) }.sum)
  }

  def findKNearestNeighbors(v: MLVector, allVecs: Array[MLVector], k: Int): Array[MLVector] = {
    allVecs
      .filter(_ != v)
      .map(x => (x, euclideanDist(v, x)))
      .sortBy(_._2)
      .take(k)
      .map(_._1)
  }

  def generateSyntheticSample(v: MLVector, neighbors: Array[MLVector]): MLVector = {
    val neighbor = neighbors(Random.nextInt(neighbors.length))
    val arrV = v.toArray
    val arrN = neighbor.toArray
    val t = Random.nextDouble()
    val synthetic = arrV.zip(arrN).map { case (a, b) => a + (b - a) * t }
    Vectors.dense(synthetic)
  }
}
