import org.apache.spark.SparkConf
import org.apache.spark.streaming._
import org.apache.spark.streaming.kafka010._
import org.apache.kafka.common.serialization.StringDeserializer
import scalaj.http._

object KafkaConsumerGRU {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("KafkaConsumerGRU").setMaster("local[*]")
    val ssc = new StreamingContext(conf, Seconds(10))

    val kafkaParams = Map[String, Object](
      "bootstrap.servers" -> "localhost:9092",
      "key.deserializer" -> classOf[StringDeserializer],
      "value.deserializer" -> classOf[StringDeserializer],
      "group.id" -> "gru-group",
      "auto.offset.reset" -> "latest",
      "enable.auto.commit" -> (false: java.lang.Boolean)
    )

    val topics = Array("model-input")
    val stream = KafkaUtils.createDirectStream[String, String](
      ssc,
      LocationStrategies.PreferConsistent,
      ConsumerStrategies.Subscribe[String, String](topics, kafkaParams)
    )

    stream.map(record => {
      val response = Http("http://localhost:5000/predict")
        .postData(record.value())
        .header("content-type", "application/json")
        .asString
      println(s"Tahmin Sonucu: ${response.body}")
    }).print()

    ssc.start()
    ssc.awaitTermination()
  }
}
