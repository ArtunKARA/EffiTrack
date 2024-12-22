import org.apache.kafka.clients.producer._

object KafkaProducerGRU {
  def main(args: Array[String]): Unit = {
    // Kafka Ayarları
    val props = new java.util.Properties()
    props.put("bootstrap.servers", "localhost:9092")
    props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer")
    props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer")

    // Kafka Producer
    val producer = new KafkaProducer[String, String](props)

    // Örnek Sonuç (Bunu tahmin edilen sonuçla değiştirin)
    val result = """{
      "features": [1.2, 3.4, 5.6],
      "actual_label": 1,
      "predicted_label": 0,
      "is_anomaly": true
    }"""

    // Kafka'ya Mesaj Gönderimi
    val topic = "model-output"
    val record = new ProducerRecord[String, String](topic, null, result)
    producer.send(record)
    println(s"Sonuç Kafka'ya gönderildi: $result")

    producer.close()
  }
}
