from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, datediff, to_date, mean, stddev
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

# Initialize Spark session
spark = SparkSession.builder \
    .appName("MIMIC-II Blood Pressure Prediction") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# Database connection parameters (replace with your credentials)
db_url = "jdbc:postgresql://your_host:your_port/mimic2"
properties = {
    "user": "your_username",
    "password": "your_password",
    "driver": "org.postgresql.Driver"
}

# Query relevant data from MIMIC-II
def query_bp_data():
    query = """
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
        c.itemid IN (
            -- Systolic BP
            6, 51, 455, 6701, 220050, 220179,
            -- Diastolic BP
            8364, 8368, 8440, 8441, 8555, 220051, 220180,
            -- Mean Arterial Pressure
            52, 456, 6702, 220052, 220181,
            -- Heart Rate
            211, 220045
        )
        AND c.error IS NULL
        AND c.valuenum IS NOT NULL
    ORDER BY 
        a.hadm_id, c.charttime
    LIMIT 100000) as bp_data
    """
    
    try:
        df = spark.read.jdbc(url=db_url, table=query, properties=properties)
        print("Data queried successfully")
        return df
    except Exception as e:
        print(f"Error querying data: {e}")
        return None

# Preprocess the data
def preprocess_data(df):
    # Define measurement types
    systolic_bp_items = [6, 51, 455, 6701, 220050, 220179]
    diastolic_bp_items = [8364, 8368, 8440, 8441, 8555, 220051, 220180]
    map_items = [52, 456, 6702, 220052, 220181]
    hr_items = [211, 220045]
    
    # Create measurement type column
    df = df.withColumn("measurement",
        when(col("itemid").isin(systolic_bp_items), "Systolic BP")
        .when(col("itemid").isin(diastolic_bp_items), "Diastolic BP")
        .when(col("itemid").isin(map_items), "MAP")
        .when(col("itemid").isin(hr_items), "Heart Rate")
        .otherwise("Other")
    )
    
    # Filter only relevant measurements
    df = df.filter(col("measurement") != "Other")
    
    # Pivot the data to have measurements as columns
    pivoted_df = df.groupBy("hadm_id", "charttime") \
        .pivot("measurement", ["Systolic BP", "Diastolic BP", "MAP", "Heart Rate"]) \
        .agg(mean("valuenum").alias("value"))
    
    # Calculate age at admission
    df_demo = df.select("hadm_id", "gender", "dob", "admittime").distinct()
    df_demo = df_demo.withColumn("admittime", to_date(col("admittime")))
    df_demo = df_demo.withColumn("dob", to_date(col("dob")))
    df_demo = df_demo.withColumn("age", 
        datediff(col("admittime"), col("dob")) / 365.25)
    
    # Join demographic features
    pivoted_df = pivoted_df.join(df_demo.select("hadm_id", "gender", "age"), "hadm_id")
    
    # Convert gender to numeric
    gender_indexer = StringIndexer(inputCol="gender", outputCol="gender_index")
    
    # Drop rows with missing values in key features
    pivoted_df = pivoted_df.dropna(subset=["Systolic BP", "Diastolic BP", "Heart Rate"])
    
    return pivoted_df, gender_indexer

# Train and evaluate model
def train_and_evaluate(df, gender_indexer):
    # Feature columns
    feature_cols = ["Diastolic BP", "Heart Rate", "gender_index", "age"]
    
    # Assemble features
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features"
    )
    
    # Define model
    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol="Systolic BP",
        numTrees=100,
        maxDepth=5,
        seed=42
    )
    
    # Create pipeline
    pipeline = Pipeline(stages=[gender_indexer, assembler, rf])
    
    # Split data
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
    
    # Train model
    model = pipeline.fit(train_data)
    
    # Make predictions
    predictions = model.transform(test_data)
    
    # Evaluate model
    evaluator_mae = RegressionEvaluator(
        labelCol="Systolic BP",
        predictionCol="prediction",
        metricName="mae"
    )
    
    evaluator_rmse = RegressionEvaluator(
        labelCol="Systolic BP",
        predictionCol="prediction",
        metricName="rmse"
    )
    
    evaluator_r2 = RegressionEvaluator(
        labelCol="Systolic BP",
        predictionCol="prediction",
        metricName="r2"
    )
    
    mae = evaluator_mae.evaluate(predictions)
    rmse = evaluator_rmse.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)
    
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R2): {r2:.2f}")
    
    return model, predictions

# Main execution
if __name__ == "__main__":
    # Query data
    df = query_bp_data()
    
    if df:
        # Preprocess data
        pivoted_df, gender_indexer = preprocess_data(df)
        
        # Cache the DataFrame as we'll use it multiple times
        pivoted_df.cache()
        
        # Show schema and sample data
        pivoted_df.printSchema()
        pivoted_df.show(5)
        
        # Train and evaluate model
        model, predictions = train_and_evaluate(pivoted_df, gender_indexer)
        
        # Show some predictions
        predictions.select("Systolic BP", "prediction").show(10)
        
        # Unpersist the DataFrame
        pivoted_df.unpersist()
    
    # Stop Spark session
    spark.stop()