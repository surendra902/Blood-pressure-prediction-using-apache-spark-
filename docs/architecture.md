# System Architecture

## Overview

The Blood Pressure Prediction system consists of two main pipelines:

1. **Batch Pipeline** — Processes historical MIMIC-II clinical data for model training and offline prediction.
2. **Streaming Pipeline** — Ingests real-time sensor data via Apache Kafka for continuous BP monitoring.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        BATCH PIPELINE                                   │
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │  MIMIC-II DB │───►│  Apache Spark│───►│  Spark MLlib │              │
│  │ (PostgreSQL) │    │   (PySpark)  │    │ Random Forest│              │
│  │              │    │              │    │              │              │
│  │ - patients   │    │ - Query JDBC │    │ - Train model│              │
│  │ - admissions │    │ - Clean data │    │ - Evaluate   │              │
│  │ - chartevents│    │ - Pivot cols  │    │ - Predict    │              │
│  │ - d_items    │    │ - Feature eng│    │ - Export CSV │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                      STREAMING PIPELINE                                 │
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │  Wearable    │───►│ Apache Kafka │───►│ Spark        │              │
│  │  Sensors     │    │              │    │ Structured   │              │
│  │              │    │ Topic:       │    │ Streaming    │              │
│  │ - PTT        │    │  bp-stream   │    │              │              │
│  │ - PIR        │    │              │    │ foreachBatch │              │
│  │ - HR         │    │              │    │ ─► Predict   │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
└─────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Batch Pipeline

1. **Ingestion**: JDBC connection to MIMIC-II PostgreSQL database
2. **Extraction**: Query patients, admissions, chart events, and item definitions
3. **Transformation**:
   - Classify measurements (Systolic BP, Diastolic BP, MAP, Heart Rate)
   - Pivot measurement types into separate columns
   - Calculate patient age from date of birth and admission time
   - Encode categorical gender feature via StringIndexer
4. **Training**: Random Forest Regressor (100 trees, max depth 5)
5. **Evaluation**: MAE, RMSE, R² metrics on 20% holdout test set
6. **Output**: Predictions exported as CSV

### Streaming Pipeline

1. **Ingestion**: Kafka consumer reads JSON messages from `bp-stream` topic
2. **Parsing**: Structured Streaming parses JSON into typed columns
3. **Feature Assembly**: VectorAssembler combines PTT, PIR, HR
4. **Prediction**: Pre-trained models predict SBP and DBP per micro-batch
5. **Output**: Real-time predictions logged to console (extensible to sinks)

## Technology Stack

| Component        | Technology                     | Version    |
|------------------|--------------------------------|------------|
| Processing       | Apache Spark (PySpark)         | ≥ 3.4.0    |
| ML Framework     | Spark MLlib                    | (bundled)  |
| Streaming        | Spark Structured Streaming     | (bundled)  |
| Message Queue    | Apache Kafka                   | ≥ 3.0.0    |
| Database         | PostgreSQL (MIMIC-II)          | ≥ 12.0     |
| Language         | Python                         | ≥ 3.9      |
| JDBC Driver      | PostgreSQL JDBC                | 42.2.24    |

## Model Details

### Algorithm: Random Forest Regressor

- **Ensemble method**: Bagging with decision trees
- **Number of trees**: 100 (configurable)
- **Max depth**: 5 (configurable)
- **Seed**: 42 (reproducible results)

### Features

| Feature          | Source          | Description                            |
|------------------|-----------------|----------------------------------------|
| Diastolic BP     | MIMIC-II        | Diastolic blood pressure reading       |
| Heart Rate       | MIMIC-II        | Heart rate measurement                 |
| Gender (indexed) | MIMIC-II        | Patient gender (StringIndexed)         |
| Age              | Calculated      | Age at admission (years)               |

### Target Variable

- **Systolic Blood Pressure (SBP)** — Predicted in mmHg
