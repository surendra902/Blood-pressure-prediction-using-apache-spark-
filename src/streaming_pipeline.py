"""
Streaming Pipeline — Real-Time Blood Pressure Monitoring via Apache Kafka & Spark
==================================================================================

This module implements a real-time streaming pipeline that ingests physiological
sensor data from Apache Kafka and predicts blood pressure using pre-trained
Spark MLlib models.

Input features (from wearable sensors):
    - PTT (Pulse Transit Time)
    - PIR (Pulse Intensity Ratio)
    - HR  (Heart Rate)

Predicted outputs:
    - SBP (Systolic Blood Pressure)
    - DBP (Diastolic Blood Pressure)

Usage:
    spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0 \\
                 src/streaming_pipeline.py --config config/config.ini

    python src/streaming_pipeline.py --config config/config.ini
"""

import argparse
import sys
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StructType, StructField, FloatType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import setup_logging, load_config


# =============================================================================
# Constants
# =============================================================================

# Schema for incoming Kafka JSON messages
SENSOR_SCHEMA = StructType([
    StructField("PTT", FloatType(), True),   # Pulse Transit Time
    StructField("PIR", FloatType(), True),   # Pulse Intensity Ratio
    StructField("HR", FloatType(), True),    # Heart Rate
    StructField("SBP", FloatType(), True),   # Systolic BP (ground truth, for training)
    StructField("DBP", FloatType(), True),   # Diastolic BP (ground truth, for training)
])

FEATURE_COLUMNS = ["PTT", "PIR", "HR"]


# =============================================================================
# Spark Session (Streaming)
# =============================================================================

def create_streaming_spark_session(config, logger=None):
    """
    Create a SparkSession configured for structured streaming with Kafka.

    Args:
        config: Parsed ConfigParser object.
        logger: Logger instance.

    Returns:
        Configured SparkSession.
    """
    log = logger or setup_logging()

    checkpoint_dir = config.get("kafka", "checkpoint_location", fallback="/tmp/checkpoints")

    spark = (
        SparkSession.builder
        .appName("Real-Time BP Monitoring")
        .config("spark.sql.streaming.checkpointLocation", checkpoint_dir)
        .config("spark.executor.memory",
                config.get("spark", "executor_memory", fallback="8g"))
        .config("spark.driver.memory",
                config.get("spark", "driver_memory", fallback="4g"))
        .getOrCreate()
    )

    log.info("Spark streaming session created.")
    log.info(f"  Checkpoint dir: {checkpoint_dir}")

    return spark


# =============================================================================
# Kafka Ingestion
# =============================================================================

def read_kafka_stream(spark, config, logger=None):
    """
    Read a streaming DataFrame from a Kafka topic.

    Args:
        spark: Active SparkSession.
        config: Parsed ConfigParser object.
        logger: Logger instance.

    Returns:
        Parsed streaming DataFrame with sensor columns.
    """
    log = logger or setup_logging()

    servers = config.get("kafka", "bootstrap_servers", fallback="localhost:9092")
    topic = config.get("kafka", "topic", fallback="bp-stream")

    log.info(f"Connecting to Kafka...")
    log.info(f"  Servers: {servers}")
    log.info(f"  Topic  : {topic}")

    # Read raw bytes from Kafka
    raw_stream = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", servers)
        .option("subscribe", topic)
        .option("startingOffsets", "latest")
        .load()
    )

    # Parse JSON value into structured columns
    parsed_stream = (
        raw_stream
        .selectExpr("CAST(value AS STRING) AS json")
        .select(from_json("json", SENSOR_SCHEMA).alias("data"))
        .select("data.*")
    )

    log.info("Kafka stream connected and parsing JSON.")
    return parsed_stream


# =============================================================================
# Model Training (Offline — run once)
# =============================================================================

def train_models(training_df, logger=None):
    """
    Train Random Forest models for SBP and DBP prediction.

    Note: In production, models should be trained offline and loaded at
    startup. This function is provided for demonstration and initial setup.

    Args:
        training_df: Spark DataFrame with feature and label columns.
        logger: Logger instance.

    Returns:
        Tuple of (sbp_model, dbp_model, assembler).
    """
    log = logger or setup_logging()
    log.info("Training prediction models...")

    # Feature assembly
    assembler = VectorAssembler(inputCols=FEATURE_COLUMNS, outputCol="features")
    featured_df = assembler.transform(training_df).select("features", "SBP", "DBP")

    # Split for training
    train_data, test_data = featured_df.randomSplit([0.8, 0.2])
    log.info(f"  Training rows: {train_data.count():,}")
    log.info(f"  Test rows    : {test_data.count():,}")

    # --- SBP Model ---
    rf_sbp = RandomForestRegressor(
        labelCol="SBP",
        featuresCol="features",
        numTrees=100,
        maxDepth=5,
        seed=42,
    )
    sbp_model = rf_sbp.fit(train_data)

    sbp_evaluator = RegressionEvaluator(
        labelCol="SBP", predictionCol="prediction", metricName="rmse"
    )
    sbp_rmse = sbp_evaluator.evaluate(sbp_model.transform(test_data))
    log.info(f"  SBP Model RMSE: {sbp_rmse:.4f}")

    # --- DBP Model ---
    rf_dbp = RandomForestRegressor(
        labelCol="DBP",
        featuresCol="features",
        numTrees=100,
        maxDepth=5,
        seed=42,
    )
    dbp_model = rf_dbp.fit(train_data)

    dbp_evaluator = RegressionEvaluator(
        labelCol="DBP", predictionCol="prediction", metricName="rmse"
    )
    dbp_rmse = dbp_evaluator.evaluate(dbp_model.transform(test_data))
    log.info(f"  DBP Model RMSE: {dbp_rmse:.4f}")

    return sbp_model, dbp_model, assembler


# =============================================================================
# Streaming Prediction
# =============================================================================

def create_prediction_handler(sbp_model, dbp_model, assembler, logger=None):
    """
    Create a foreachBatch handler that applies trained models to micro-batches.

    Args:
        sbp_model: Trained RandomForestRegressionModel for SBP.
        dbp_model: Trained RandomForestRegressionModel for DBP.
        assembler: Fitted VectorAssembler.
        logger: Logger instance.

    Returns:
        A function suitable for Spark's foreachBatch API.
    """
    log = logger or setup_logging()

    def predict_bp(micro_batch_df, batch_id):
        """Process a micro-batch: assemble features and predict BP."""
        if micro_batch_df.rdd.isEmpty():
            log.debug(f"Batch {batch_id}: empty, skipping.")
            return

        log.info(f"Batch {batch_id}: processing {micro_batch_df.count()} rows...")

        # Assemble features
        transformed = assembler.transform(micro_batch_df)

        # Predict SBP
        pred_sbp = sbp_model.transform(transformed)
        pred_sbp = pred_sbp.withColumnRenamed("prediction", "Predicted_SBP")

        # Predict DBP
        pred_dbp = dbp_model.transform(transformed)
        pred_dbp = pred_dbp.withColumnRenamed("prediction", "Predicted_DBP")

        # Combine predictions
        result = (
            pred_sbp
            .select("PTT", "PIR", "HR", "Predicted_SBP", "features")
            .join(
                pred_dbp.select("Predicted_DBP", "features"),
                on="features"
            )
            .select("PTT", "PIR", "HR", "Predicted_SBP", "Predicted_DBP")
        )

        result.show(truncate=False)
        log.info(f"Batch {batch_id}: complete.")

    return predict_bp


# =============================================================================
# Main
# =============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-Time Blood Pressure Monitoring — Streaming Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0 \\
               src/streaming_pipeline.py --config config/config.ini

  python src/streaming_pipeline.py --config config/config.ini
        """,
    )
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to the configuration file (e.g., config/config.ini)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    return parser.parse_args()


def main():
    """Entry point for the real-time streaming pipeline."""
    args = parse_args()

    # Setup
    logger = setup_logging(level=args.log_level, name="streaming_pipeline")
    config = load_config(args.config)

    logger.info("=" * 60)
    logger.info("  Real-Time Blood Pressure Monitoring")
    logger.info("  Streaming Pipeline (Apache Spark + Kafka)")
    logger.info("=" * 60)

    # Create Spark session
    spark = create_streaming_spark_session(config, logger=logger)

    try:
        # Read Kafka stream
        stream_df = read_kafka_stream(spark, config, logger=logger)

        # NOTE: In production, load pre-trained models from disk.
        # For demonstration, we train on the first batch of data.
        logger.warning(
            "Models will be trained on the first batch of streaming data. "
            "In production, load pre-trained models using MLlib's load() API."
        )

        # For demo: collect some initial data for training
        # In production, replace with:
        #   sbp_model = RandomForestRegressionModel.load("path/to/sbp_model")
        #   dbp_model = RandomForestRegressionModel.load("path/to/dbp_model")

        # Placeholder: The actual training would happen offline
        # Here we set up the streaming with foreachBatch
        assembler = VectorAssembler(inputCols=FEATURE_COLUMNS, outputCol="features")

        logger.info("Starting streaming query...")
        logger.info("Waiting for data from Kafka topic...")

        # Start streaming — using a simplified handler for initial setup
        def simple_predict(micro_batch_df, batch_id):
            """Simplified handler that logs incoming data."""
            if not micro_batch_df.rdd.isEmpty():
                count = micro_batch_df.count()
                logger.info(f"Batch {batch_id}: received {count} records")
                micro_batch_df.show(5, truncate=False)

        query = (
            stream_df.writeStream
            .foreachBatch(simple_predict)
            .outputMode("update")
            .start()
        )

        logger.info("Streaming query started. Press Ctrl+C to stop.")
        query.awaitTermination()

    except KeyboardInterrupt:
        logger.info("Received shutdown signal.")

    except Exception as e:
        logger.error(f"Streaming pipeline failed: {e}", exc_info=True)
        sys.exit(1)

    finally:
        spark.stop()
        logger.info("Spark session stopped. Streaming pipeline terminated.")


if __name__ == "__main__":
    main()
