# EffiTrack

## Project Datas

https://www.kaggle.com/datasets/inIT-OWL/high-storage-system-data-for-energy-optimization?resource=download&select=HRSS_anomalous_optimized.csv

## Start Real Time Proces
### Start Env
zkserver
kafka-server-start.bat %KAFKA_HOME%\config\server.properties
### Kafka Topic Oluşturma
kafka-topics.bat --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic anomalies
kafka-topics.bat --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic normal_data
###  Veri Kafka'ya Gönderildikten Sonra Kontrol Etme
#### Anomaliler İçin
kafka-console-consumer.bat --bootstrap-server localhost:9092 --topic anomalies --from-beginning
#### Normal Veriler İçin
kafka-console-consumer.bat --bootstrap-server localhost:9092 --topic normal_data --from-beginning
