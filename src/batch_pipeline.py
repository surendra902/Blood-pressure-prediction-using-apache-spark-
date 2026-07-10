"""
Batch Pipeline — Blood Pressure Prediction using MIMIC-II & Apache Spark MLlib
===============================================================================

This module implements the batch processing pipeline for predicting systolic
blood pressure from clinical data stored in the MIMIC-II database.

Pipeline stages:
    1. Data Ingestion  — Query MIMIC-II via JDBC
    2. Preprocessing   — Clean, pivot, and engineer features
    3. Model Training  — Random Forest regression via Spark MLlib
    4. Evaluation      — MAE, RMSE, R² metrics
    5. Feature Analysis — Feature importance ranking

Usage:
    python src/batch_pipeline.py --config config/config.ini
    python src/batch_pipeline.py --config config/config.ini --output results/predictions.csv

    spark-submit --driver-class-path ./lib/postgresql-42.2.24.jar \\
                 --jars ./lib/postgresql-42.2.24.jar \\
                 src/batch_pipeline.py --config config/config.ini
"""

import argparse
import sys
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, datediff, to_date, mean
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    setup_logging,
    load_config,
    create_spark_session,
    get_db_properties,
    get_jdbc_url,
    get_item_ids,
    validate_dataframe,
    print_summary,
)


# =============================================================================
# Constants
# =============================================================================

FEATURE_COLUMNS = ["Diastolic BP", "Heart Rate", "gender_index", "age"]
LABEL_COLUMN = "Systolic BP"


# =============================================================================
# Data Ingestion
# =============================================================================

def query_bp_data(spark, db_url, properties, item_ids, limit=100000, logger=None):
    """
    Query blood pressure and related vital sign data from MIMIC-II.

    Retrieves patient demographics, admission records, and chart events
    for systolic BP, diastolic BP, mean arterial pressure, and heart rate.

    Args:
        spark: Active SparkSession.
        db_url: JDBC connection URL.
        properties: JDBC connection properties dict.
        item_ids: Dict mapping measurement types to item ID lists.
        limit: Maximum number of rows to fetch.
        logger: Logger instance.

    Returns:
        Spark DataFrame with raw clinical data, or None on failure.
    """
    log = logger or setup_logging()

    # Flatten all item IDs into a single list for the SQL IN clause
    all_items = (
        item_ids["systolic"] + item_ids["diastolic"] +
        item_ids["map"] + item_ids["hr"]
    )
    items_str = ", ".join(str(i) for i in all_items)

    query = f"""
    (SELECT
        p.gender, p.dob, p.dod,
        a.hadm_id, a.admittime, a.dischtime,
        c.itemid, c.charttime, c.value, c.valuenum, c.valueuom, c.error,
        d.label, d.unitname
    FROM
        patients p
    JOIN
        admissions a ON p.subject_id = a.subject_id
    JOIN
        chartevents c ON a.hadm_id = c.hadm_id
    JOIN
        d_items d ON c.itemid = d.itemid
    WHERE
        c.itemid IN ({items_str})
        AND c.error IS NULL
        AND c.valuenum IS NOT NULL
    ORDER BY
        a.hadm_id, c.charttime
    LIMIT {limit}) AS bp_data
    """

    try:
        log.info("Querying MIMIC-II database...")
        log.info(f"  JDBC URL : {db_url}")
        log.info(f"  Row limit: {limit:,}")

        df = spark.read.jdbc(url=db_url, table=query, properties=properties)

        row_count = df.count()
        log.info(f"  Retrieved: {row_count:,} rows")
        return df

    except Exception as e:
        log.error(f"Failed to query data: {e}")
        return None


# =============================================================================
# Preprocessing
# =============================================================================

def preprocess_data(df, item_ids, logger=None):
    """
    Clean, transform, and engineer features from raw clinical data.

    Steps:
        1. Classify each row by measurement type (Systolic BP, etc.)
        2. Pivot measurements into columns
        3. Calculate patient age at admission
        4. Index categorical gender column
        5. Drop rows with missing key features

    Args:
        df: Raw Spark DataFrame from query_bp_data().
        item_ids: Dict mapping measurement types to item ID lists.
        logger: Logger instance.

    Returns:
        Tuple of (processed DataFrame, StringIndexer for gender).
    """
    log = logger or setup_logging()
    log.info("Preprocessing data...")

    # --- Step 1: Classify measurement types ---
    df = df.withColumn(
        "measurement",
        when(col("itemid").isin(item_ids["systolic"]), "Systolic BP")
        .when(col("itemid").isin(item_ids["diastolic"]), "Diastolic BP")
        .when(col("itemid").isin(item_ids["map"]), "MAP")
        .when(col("itemid").isin(item_ids["hr"]), "Heart Rate")
        .otherwise("Other")
    )

    df = df.filter(col("measurement") != "Other")
    log.info(f"  After measurement classification: {df.count():,} rows")

    # --- Step 2: Pivot measurements into columns ---
    pivoted_df = (
        df.groupBy("hadm_id", "charttime")
        .pivot("measurement", ["Systolic BP", "Diastolic BP", "MAP", "Heart Rate"])
        .agg(mean("valuenum").alias("value"))
    )
    log.info(f"  After pivot: {pivoted_df.count():,} rows")

    # --- Step 3: Calculate age at admission ---
    df_demo = df.select("hadm_id", "gender", "dob", "admittime").distinct()
    df_demo = df_demo.withColumn("admittime", to_date(col("admittime")))
    df_demo = df_demo.withColumn("dob", to_date(col("dob")))
    df_demo = df_demo.withColumn(
        "age",
        datediff(col("admittime"), col("dob")) / 365.25
    )

    # --- Step 4: Join demographics ---
    pivoted_df = pivoted_df.join(
        df_demo.select("hadm_id", "gender", "age"),
        on="hadm_id"
    )

    # --- Step 5: Gender indexing ---
    gender_indexer = StringIndexer(inputCol="gender", outputCol="gender_index")

    # --- Step 6: Drop nulls in key features ---
    pivoted_df = pivoted_df.dropna(subset=["Systolic BP", "Diastolic BP", "Heart Rate"])
    log.info(f"  After dropping nulls: {pivoted_df.count():,} rows")

    return pivoted_df, gender_indexer


# =============================================================================
# Model Training & Evaluation
# =============================================================================

def train_and_evaluate(df, gender_indexer, config=None, logger=None):
    """
    Train a Random Forest model and evaluate performance.

    Args:
        df: Preprocessed Spark DataFrame.
        gender_indexer: Fitted or unfitted StringIndexer for gender.
        config: Optional ConfigParser for hyperparameters.
        logger: Logger instance.

    Returns:
        Tuple of (trained PipelineModel, predictions DataFrame, metrics dict).
    """
    log = logger or setup_logging()

    # --- Hyperparameters ---
    num_trees = int(config.get("model", "num_trees", fallback="100")) if config else 100
    max_depth = int(config.get("model", "max_depth", fallback="5")) if config else 5
    seed = int(config.get("model", "seed", fallback="42")) if config else 42
    train_ratio = float(config.get("model", "train_split", fallback="0.8")) if config else 0.8

    log.info("Training Random Forest model...")
    log.info(f"  Hyperparameters: num_trees={num_trees}, max_depth={max_depth}, seed={seed}")
    log.info(f"  Train/Test split: {train_ratio}/{1 - train_ratio}")

    # --- Assemble features ---
    assembler = VectorAssembler(
        inputCols=FEATURE_COLUMNS,
        outputCol="features"
    )

    # --- Define model ---
    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol=LABEL_COLUMN,
        numTrees=num_trees,
        maxDepth=max_depth,
        seed=seed,
    )

    # --- Build & fit pipeline ---
    pipeline = Pipeline(stages=[gender_indexer, assembler, rf])
    train_data, test_data = df.randomSplit([train_ratio, 1 - train_ratio], seed=seed)

    log.info(f"  Training set : {train_data.count():,} rows")
    log.info(f"  Test set     : {test_data.count():,} rows")

    model = pipeline.fit(train_data)
    predictions = model.transform(test_data)

    # --- Evaluate ---
    metrics = {}
    for metric_name in ["mae", "rmse", "r2"]:
        evaluator = RegressionEvaluator(
            labelCol=LABEL_COLUMN,
            predictionCol="prediction",
            metricName=metric_name,
        )
        metrics[metric_name] = evaluator.evaluate(predictions)

    log.info("=" * 50)
    log.info("  MODEL EVALUATION RESULTS")
    log.info("=" * 50)
    log.info(f"  Mean Absolute Error (MAE) : {metrics['mae']:.4f}")
    log.info(f"  Root Mean Squared Error   : {metrics['rmse']:.4f}")
    log.info(f"  R-squared (R²)            : {metrics['r2']:.4f}")
    log.info("=" * 50)

    # --- Feature Importance ---
    rf_model = model.stages[-1]  # Extract the RF model from the pipeline
    importances = rf_model.featureImportances.toArray()

    log.info("  Feature Importances:")
    for feat, imp in sorted(zip(FEATURE_COLUMNS, importances), key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        log.info(f"    {feat:<16s} : {imp:.4f}  {bar}")

    return model, predictions, metrics


# =============================================================================
# Main
# =============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Blood Pressure Prediction — Batch Pipeline (Apache Spark)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/batch_pipeline.py --config config/config.ini
  python src/batch_pipeline.py --config config/config.ini --output results/predictions.csv
  spark-submit --jars ./lib/postgresql-42.2.24.jar src/batch_pipeline.py --config config/config.ini
        """,
    )
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to the configuration file (e.g., config/config.ini)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path to save predictions CSV (optional)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    return parser.parse_args()


def main():
    """Entry point for the batch prediction pipeline."""
    args = parse_args()

    # Setup
    logger = setup_logging(level=args.log_level, name="batch_pipeline")
    config = load_config(args.config)

    logger.info("=" * 60)
    logger.info("  Blood Pressure Prediction — Batch Pipeline")
    logger.info("=" * 60)

    # Create Spark session
    spark = create_spark_session(config, app_name="MIMIC-II BP Prediction (Batch)")

    try:
        # 1. Ingest
        db_url = get_jdbc_url(config)
        db_props = get_db_properties(config)
        item_ids = get_item_ids(config)
        limit = int(config.get("data", "query_limit", fallback="100000"))

        df = query_bp_data(spark, db_url, db_props, item_ids, limit=limit, logger=logger)

        if df is None:
            logger.error("Data ingestion failed. Exiting.")
            sys.exit(1)

        # 2. Preprocess
        processed_df, gender_indexer = preprocess_data(df, item_ids, logger=logger)
        processed_df.cache()
        print_summary(processed_df, name="Processed Data", logger=logger)

        # 3. Train & Evaluate
        model, predictions, metrics = train_and_evaluate(
            processed_df, gender_indexer, config=config, logger=logger
        )

        # 4. Show sample predictions
        logger.info("Sample predictions:")
        predictions.select(LABEL_COLUMN, "prediction").show(10)

        # 5. Save output (optional)
        if args.output:
            logger.info(f"Saving predictions to: {args.output}")
            predictions.select(LABEL_COLUMN, "prediction").coalesce(1).write.csv(
                args.output, header=True, mode="overwrite"
            )
            logger.info("Predictions saved successfully.")

        # Cleanup
        processed_df.unpersist()

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)

    finally:
        spark.stop()
        logger.info("Spark session stopped. Pipeline complete.")


if __name__ == "__main__":
    main()
