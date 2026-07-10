"""
Utility Module — Shared helpers for Blood Pressure Prediction pipelines.
========================================================================

Provides:
    - Spark session builder with configurable resources
    - Configuration file loader and validator
    - Data quality validators
    - Logging setup
"""

import logging
import os
import sys
import configparser
from typing import Optional, Dict, List


# =============================================================================
# Logging
# =============================================================================

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    name: str = "bp_prediction"
) -> logging.Logger:
    """
    Configure and return a logger instance.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to a log file. If None, logs to console only.
        name: Logger name.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# =============================================================================
# Configuration
# =============================================================================

def load_config(config_path: str) -> configparser.ConfigParser:
    """
    Load and validate a configuration file.

    Args:
        config_path: Absolute or relative path to the .ini config file.

    Returns:
        Parsed ConfigParser object.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If required sections are missing.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Copy config/config.ini.template to config/config.ini and fill in your values."
        )

    config = configparser.ConfigParser()
    config.read(config_path)

    required_sections = ["database", "spark"]
    missing = [s for s in required_sections if s not in config.sections()]
    if missing:
        raise ValueError(
            f"Configuration file is missing required sections: {missing}"
        )

    return config


def get_db_properties(config: configparser.ConfigParser) -> Dict[str, str]:
    """
    Extract database connection properties from config.

    Args:
        config: Parsed ConfigParser object.

    Returns:
        Dictionary with JDBC connection properties.
    """
    return {
        "user": config.get("database", "user"),
        "password": config.get("database", "password"),
        "driver": "org.postgresql.Driver"
    }


def get_jdbc_url(config: configparser.ConfigParser) -> str:
    """
    Build the JDBC URL from config parameters.

    Args:
        config: Parsed ConfigParser object.

    Returns:
        JDBC connection URL string.
    """
    host = config.get("database", "host")
    port = config.get("database", "port")
    dbname = config.get("database", "dbname")
    return f"jdbc:postgresql://{host}:{port}/{dbname}"


def get_item_ids(config: configparser.ConfigParser) -> Dict[str, List[int]]:
    """
    Parse measurement item IDs from config.

    Args:
        config: Parsed ConfigParser object.

    Returns:
        Dictionary mapping measurement types to lists of item IDs.
    """
    return {
        "systolic": [int(x.strip()) for x in config.get("data", "systolic_items").split(",")],
        "diastolic": [int(x.strip()) for x in config.get("data", "diastolic_items").split(",")],
        "map": [int(x.strip()) for x in config.get("data", "map_items").split(",")],
        "hr": [int(x.strip()) for x in config.get("data", "hr_items").split(",")],
    }


# =============================================================================
# Spark Session
# =============================================================================

def create_spark_session(
    config: configparser.ConfigParser,
    app_name: Optional[str] = None
):
    """
    Create and return a configured SparkSession.

    Args:
        config: Parsed ConfigParser object.
        app_name: Optional override for the Spark application name.

    Returns:
        Configured SparkSession instance.
    """
    from pyspark.sql import SparkSession

    name = app_name or config.get("spark", "app_name", fallback="BP Prediction")
    executor_mem = config.get("spark", "executor_memory", fallback="8g")
    driver_mem = config.get("spark", "driver_memory", fallback="4g")
    max_result = config.get("spark", "max_result_size", fallback="4g")

    spark = (
        SparkSession.builder
        .appName(name)
        .config("spark.executor.memory", executor_mem)
        .config("spark.driver.memory", driver_mem)
        .config("spark.driver.maxResultSize", max_result)
        .getOrCreate()
    )

    return spark


# =============================================================================
# Data Validation
# =============================================================================

def validate_dataframe(df, required_columns: List[str], name: str = "DataFrame"):
    """
    Validate that a Spark DataFrame contains required columns and is non-empty.

    Args:
        df: Spark DataFrame to validate.
        required_columns: List of column names that must be present.
        name: Human-readable name for error messages.

    Raises:
        ValueError: If required columns are missing or DataFrame is empty.
    """
    if df is None:
        raise ValueError(f"{name} is None — data query may have failed.")

    existing_cols = set(df.columns)
    missing_cols = [c for c in required_columns if c not in existing_cols]

    if missing_cols:
        raise ValueError(
            f"{name} is missing required columns: {missing_cols}\n"
            f"Available columns: {list(existing_cols)}"
        )

    count = df.count()
    if count == 0:
        raise ValueError(f"{name} is empty (0 rows). Check your query and filters.")

    return count


def print_summary(df, name: str = "DataFrame", logger: Optional[logging.Logger] = None):
    """
    Print a summary of a Spark DataFrame.

    Args:
        df: Spark DataFrame.
        name: Human-readable name.
        logger: Optional logger instance.
    """
    log = logger.info if logger else print

    log(f"{'=' * 60}")
    log(f"  {name} Summary")
    log(f"{'=' * 60}")
    log(f"  Rows   : {df.count():,}")
    log(f"  Columns: {len(df.columns)}")
    log(f"  Schema : {', '.join(df.columns)}")
    log(f"{'=' * 60}")
