# Apache Spark Cluster Setup Guide

> This document provides step-by-step instructions for setting up an Apache Spark cluster
> for the Blood Pressure Prediction system.

## Prerequisites

- **Operating System**: Ubuntu 20.04+ / CentOS 7+ / Windows 10+ with WSL
- **Java**: JDK 8 or JDK 11 (required by Spark)
- **Python**: 3.9+
- **RAM**: Minimum 8 GB per node (16 GB recommended)
- **Storage**: 50 GB+ free disk space

## 1. Install Java

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install openjdk-11-jdk -y

# CentOS/RHEL
sudo yum install java-11-openjdk-devel -y

# Verify
java -version
```

## 2. Install Apache Spark

```bash
# Download Spark 3.4.x
wget https://archive.apache.org/dist/spark/spark-3.4.1/spark-3.4.1-bin-hadoop3.tgz

# Extract
tar -xzf spark-3.4.1-bin-hadoop3.tgz
sudo mv spark-3.4.1-bin-hadoop3 /opt/spark

# Set environment variables
echo 'export SPARK_HOME=/opt/spark' >> ~/.bashrc
echo 'export PATH=$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH' >> ~/.bashrc
echo 'export PYSPARK_PYTHON=python3' >> ~/.bashrc
source ~/.bashrc

# Verify
spark-shell --version
```

## 3. Configure Spark Cluster

### 3.1 Master Node Configuration

Edit `$SPARK_HOME/conf/spark-env.sh`:

```bash
cp $SPARK_HOME/conf/spark-env.sh.template $SPARK_HOME/conf/spark-env.sh

# Add the following lines:
export SPARK_MASTER_HOST='<master-ip>'
export SPARK_MASTER_PORT=7077
export SPARK_MASTER_WEBUI_PORT=8080
export SPARK_WORKER_MEMORY=8g
export SPARK_WORKER_CORES=4
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
```

### 3.2 Worker Node Configuration

Edit `$SPARK_HOME/conf/workers`:

```
# List worker node hostnames/IPs (one per line)
worker-node-1
worker-node-2
worker-node-3
```

### 3.3 Start the Cluster

```bash
# On master node
$SPARK_HOME/sbin/start-master.sh

# On each worker node (or from master if SSH is configured)
$SPARK_HOME/sbin/start-workers.sh

# Or start all at once from master
$SPARK_HOME/sbin/start-all.sh
```

### 3.4 Verify Cluster

Open the Spark Master Web UI: `http://<master-ip>:8080`

You should see:
- Master status: ALIVE
- Worker nodes listed with their resources
- Running applications

## 4. Install PostgreSQL JDBC Driver

```bash
# Create lib directory
mkdir -p ./lib

# Download JDBC driver
wget https://jdbc.postgresql.org/download/postgresql-42.2.24.jar -P ./lib/
```

## 5. Install Apache Kafka (for Streaming Pipeline)

```bash
# Download Kafka
wget https://archive.apache.org/dist/kafka/3.4.0/kafka_2.13-3.4.0.tgz
tar -xzf kafka_2.13-3.4.0.tgz
sudo mv kafka_2.13-3.4.0 /opt/kafka

# Start Zookeeper
/opt/kafka/bin/zookeeper-server-start.sh -daemon /opt/kafka/config/zookeeper.properties

# Start Kafka broker
/opt/kafka/bin/kafka-server-start.sh -daemon /opt/kafka/config/server.properties

# Create topic for blood pressure data
/opt/kafka/bin/kafka-topics.sh --create \
    --bootstrap-server localhost:9092 \
    --topic bp-stream \
    --partitions 3 \
    --replication-factor 1
```

## 6. Submit Spark Jobs

### Batch Pipeline

```bash
spark-submit \
    --master spark://<master-ip>:7077 \
    --driver-class-path ./lib/postgresql-42.2.24.jar \
    --jars ./lib/postgresql-42.2.24.jar \
    --executor-memory 8g \
    --driver-memory 4g \
    src/batch_pipeline.py --config config/config.ini
```

### Streaming Pipeline

```bash
spark-submit \
    --master spark://<master-ip>:7077 \
    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0 \
    --executor-memory 8g \
    --driver-memory 4g \
    src/streaming_pipeline.py --config config/config.ini
```

## 7. Cluster Monitoring

| Service        | URL                            | Description          |
|----------------|--------------------------------|----------------------|
| Spark Master   | `http://<master-ip>:8080`     | Cluster overview     |
| Spark Worker   | `http://<worker-ip>:8081`     | Worker details       |
| Spark App UI   | `http://<driver-ip>:4040`     | Running application  |
| Kafka Manager  | `http://localhost:9000`        | Kafka monitoring     |

## 8. Troubleshooting

### Common Issues

| Issue                    | Solution                                              |
|--------------------------|-------------------------------------------------------|
| Connection refused       | Check firewall rules and port accessibility           |
| Out of memory            | Increase `SPARK_WORKER_MEMORY` and executor memory    |
| JDBC driver not found    | Ensure `--jars` flag includes the correct JAR path    |
| Worker not registering   | Verify SSH connectivity and hostname resolution       |
| Kafka topic not found    | Create the topic before starting the streaming pipeline|

### Useful Commands

```bash
# Check Spark cluster status
$SPARK_HOME/sbin/start-all.sh && jps

# View Spark logs
tail -f $SPARK_HOME/logs/*.out

# Stop cluster
$SPARK_HOME/sbin/stop-all.sh

# Check Kafka topics
/opt/kafka/bin/kafka-topics.sh --list --bootstrap-server localhost:9092
```
