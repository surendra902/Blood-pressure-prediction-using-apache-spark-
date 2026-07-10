<h1 align="center">
  🩺 Blood Pressure Prediction using Apache Spark
</h1>

<p align="center">
  <strong>A scalable machine learning system for predicting blood pressure from clinical and real-time sensor data</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Apache_Spark-3.4+-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white" alt="Spark"/>
  <img src="https://img.shields.io/badge/Kafka-Streaming-231F20?style=for-the-badge&logo=apachekafka&logoColor=white" alt="Kafka"/>
  <img src="https://img.shields.io/badge/MLlib-Random_Forest-FF6F00?style=for-the-badge" alt="MLlib"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"/>
</p>

---

## 📋 Overview

This project implements a **dual-pipeline machine learning system** for blood pressure prediction:

- **Batch Pipeline** — Trains a Random Forest model on historical clinical data from the [MIMIC-II](https://physionet.org/content/mimic2wdb/) database to predict **Systolic Blood Pressure (SBP)** using patient demographics, heart rate, and diastolic BP readings.

- **Streaming Pipeline** — Ingests real-time physiological sensor data (Pulse Transit Time, Pulse Intensity Ratio, Heart Rate) via **Apache Kafka** and predicts both **SBP** and **DBP** in near real-time using Spark Structured Streaming.

---

## 🏗️ Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                      BATCH PIPELINE                           │
│                                                               │
│   MIMIC-II DB ──► PySpark ETL ──► Spark MLlib ──► Predictions │
│  (PostgreSQL)     (JDBC Query)    (Random Forest)   (CSV)     │
└───────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────┐
│                    STREAMING PIPELINE                          │
│                                                               │
│   Sensors ──► Apache Kafka ──► Spark Streaming ──► Real-time  │
│  (PTT/PIR/HR)  (bp-stream)    (foreachBatch)      Predictions │
└───────────────────────────────────────────────────────────────┘
```

> 📐 For detailed architecture diagrams, see [docs/architecture.md](docs/architecture.md).

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔄 **Dual Pipeline** | Batch training + real-time streaming prediction |
| 🌲 **Random Forest** | Ensemble model with 100 trees via Spark MLlib |
| 📊 **Feature Importance** | Automatic ranking of predictive features |
| ⚙️ **Configurable** | INI-based configuration for all parameters |
| 📝 **Structured Logging** | Production-grade logging with configurable levels |
| 🔌 **MIMIC-II Integration** | Direct JDBC connection to clinical database |
| 📡 **Kafka Streaming** | Real-time data ingestion from wearable sensors |
| 🧪 **Modular Design** | Clean separation of concerns with reusable utilities |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Processing** | Apache Spark 3.4+ (PySpark) |
| **ML Framework** | Spark MLlib (Random Forest Regressor) |
| **Streaming** | Spark Structured Streaming + Apache Kafka |
| **Database** | PostgreSQL (MIMIC-II clinical database) |
| **Language** | Python 3.9+ |
| **Config** | ConfigParser (INI files) |

---

## 📁 Project Structure

```
Blood-pressure-prediction-using-apache-spark-/
│
├── src/                          # Source code
│   ├── __init__.py               # Package initialization
│   ├── batch_pipeline.py         # Batch processing & model training
│   ├── streaming_pipeline.py     # Real-time Kafka streaming
│   └── utils.py                  # Shared utilities (logging, config, Spark)
│
├── config/                       # Configuration
│   └── config.ini.template       # Template (copy to config.ini)
│
├── docs/                         # Documentation
│   ├── architecture.md           # System architecture & data flow
│   └── cluster_setup.md          # Spark cluster setup guide
│
├── notebooks/                    # Jupyter notebooks (exploratory analysis)
├── data/                         # Processed data files
├── results/                      # Prediction outputs
│
├── .gitignore                    # Git ignore rules
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT License
└── README.md                     # This file
```

---

## 🚀 Getting Started

### Prerequisites

- **Python** 3.9+
- **Java** JDK 8 or 11 (required by Spark)
- **Apache Spark** 3.4+
- **PostgreSQL** with MIMIC-II database access
- **Apache Kafka** (for streaming pipeline only)

### 1. Clone the Repository

```bash
git clone https://github.com/surendra902/Blood-pressure-prediction-using-apache-spark-.git
cd Blood-pressure-prediction-using-apache-spark-
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download JDBC Driver

```bash
mkdir -p lib
wget https://jdbc.postgresql.org/download/postgresql-42.2.24.jar -P ./lib/
```

### 5. Configure

```bash
# Copy template
cp config/config.ini.template config/config.ini

# Edit with your database credentials
nano config/config.ini
```

---

## 📖 Usage

### Batch Pipeline

Train the model and generate predictions from MIMIC-II data:

```bash
# Using Python directly
python src/batch_pipeline.py --config config/config.ini

# With output file
python src/batch_pipeline.py --config config/config.ini --output results/predictions.csv

# Using spark-submit (recommended for clusters)
spark-submit \
    --driver-class-path ./lib/postgresql-42.2.24.jar \
    --jars ./lib/postgresql-42.2.24.jar \
    src/batch_pipeline.py --config config/config.ini
```

### Streaming Pipeline

Start real-time blood pressure monitoring:

```bash
spark-submit \
    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0 \
    src/streaming_pipeline.py --config config/config.ini
```

### Command-Line Options

```
usage: batch_pipeline.py [-h] --config CONFIG [--output OUTPUT]
                         [--log-level {DEBUG,INFO,WARNING,ERROR}]

Options:
  --config, -c     Path to configuration file (required)
  --output, -o     Path to save predictions CSV (optional)
  --log-level      Logging verbosity (default: INFO)
```

---

## 🧠 Model Details

### Algorithm

**Random Forest Regressor** — An ensemble learning method that constructs multiple decision trees during training and outputs the average prediction.

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_trees` | 100 | Number of trees in the forest |
| `max_depth` | 5 | Maximum depth of each tree |
| `seed` | 42 | Random seed for reproducibility |
| `train_split` | 0.8 | Training data proportion |

### Features (Batch Pipeline)

| Feature | Type | Source |
|---------|------|--------|
| Diastolic BP | Numeric | MIMIC-II chart events |
| Heart Rate | Numeric | MIMIC-II chart events |
| Gender | Categorical (indexed) | MIMIC-II patient data |
| Age | Numeric (calculated) | DOB → admission date |

### Target Variable

- **Systolic Blood Pressure (SBP)** — measured in mmHg

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **MAE** | Mean Absolute Error — average prediction error |
| **RMSE** | Root Mean Squared Error — penalizes large errors |
| **R²** | Coefficient of Determination — explained variance |

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/architecture.md) | System design, data flow, and tech stack |
| [Cluster Setup](docs/cluster_setup.md) | Step-by-step Spark cluster configuration |
| [Config Template](config/config.ini.template) | All configurable parameters |

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/your-feature`)
3. **Commit** your changes (`git commit -m 'Add your feature'`)
4. **Push** to the branch (`git push origin feature/your-feature`)
5. **Open** a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [MIMIC-II Clinical Database](https://physionet.org/content/mimic2wdb/) — PhysioNet
- [Apache Spark](https://spark.apache.org/) — Unified analytics engine
- [Apache Kafka](https://kafka.apache.org/) — Distributed event streaming