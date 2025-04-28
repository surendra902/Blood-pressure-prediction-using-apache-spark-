# MIMIC-II Blood Pressure Prediction Project

=== SETUP INSTRUCTIONS ===

1. Create Python environment:
   python -m venv venv
   # Linux/Mac:
   source venv/bin/activate
   # Windows:
   venv\Scripts\activate

2. Install dependencies:
   pip install pandas numpy scikit-learn matplotlib psycopg2-binary sqlalchemy jupyter
   # For PySpark:
   pip install pyspark findspark

3. Download JDBC driver:
   wget https://jdbc.postgresql.org/download/postgresql-42.2.24.jar -P ./lib/

=== CONFIGURATION ===

Create config.ini with:

[database]
host = your_db_host
port = 5432
dbname = mimic2
user = your_username
password = your_password
ssl_mode = prefer

[spark]
executor_memory = 8g
driver_memory = 4g
max_result_size = 4g

Set environment variable:
export MIMIC_DB_URL="postgresql://user:password@host:port/mimic2"

=== EXECUTION ===

1. Pandas version:
   python src/pandas_pipeline.py --config config.ini --output results/predictions.csv

2. PySpark version:
   spark-submit --driver-class-path ./lib/postgresql-42.2.24.jar \
               --jars ./lib/postgresql-42.2.24.jar \
               src/spark_pipeline.py \
               --config config.ini \
               --output results/spark_predictions

3. Jupyter Notebook:
   jupyter notebook notebooks/exploratory_analysis.ipynb

=== PROJECT STRUCTURE ===

mimic-bp-prediction/
├── data/          # Processed data
├── lib/           # JDBC drivers
├── notebooks/     # Jupyter notebooks
│   └── exploratory_analysis.ipynb
├── results/       # Output files
├── src/           # Source code
│   ├── pandas_pipeline.py
│   └── spark_pipeline.py
├── config.ini     # Configuration
└── requirements.txt

=== TROUBLESHOOTING ===

1. Connection issues:
   - Verify database credentials
   - Check firewall settings
   - Test connection with psql client

2. Memory errors:
   - Reduce sample size
   - Increase Spark memory allocation
   - Use smaller batch sizes

3. Dependency conflicts:
   - Use exact versions from requirements.txt
   - Create fresh virtual environment
