ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.11.8"

lazy val root = (project in file("."))
  .settings(
    name := "SparkKafkaStreaming"
  )

// Apache Spark Core
libraryDependencies += "org.apache.spark" %% "spark-core" % "2.3.1"

// Apache Spark SQL
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.3.1"

// Apache Spark MLlib
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.3.1"

// Spark Streaming
libraryDependencies += "org.apache.spark" %% "spark-streaming" % "2.3.1"

// Spark Streaming Kafka Bağlayıcı
libraryDependencies += "org.apache.spark" %% "spark-streaming-kafka-0-10" % "2.3.1"

// Spark SQL Kafka Bağlayıcı
libraryDependencies += "org.apache.spark" %% "spark-sql-kafka-0-10" % "2.3.1"

// Kafka Clients
libraryDependencies += "org.apache.kafka" % "kafka-clients" % "2.0.0"

// Kryo (Serileştirme)
libraryDependencies += "com.esotericsoftware" % "kryo" % "4.0.2"

// Plotly (Grafik Çizimi)
libraryDependencies += "org.plotly-scala" %% "plotly-almond" % "0.5.2"

// Breeze (Matematiksel İşlemler)
libraryDependencies += "org.scalanlp" %% "breeze" % "0.13.2"

// Scalaj HTTP (HTTP istekleri için)
libraryDependencies += "org.scalaj" %% "scalaj-http" % "2.4.2"
