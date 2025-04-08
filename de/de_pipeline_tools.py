from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from delta.tables import DeltaTable
from delta import configure_spark_with_delta_pip
import time
import logging
import os
from typing import List, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Spark session with Delta Lake support
def initialize_spark_delta_lake(app_name="DE Pipeline"):
    """
    Initialize a Spark session with Delta Lake support for local execution
    
    Parameters:
    - app_name: Name of the Spark application
    
    Returns:
    - SparkSession: Configured Spark session with Delta Lake support
    """
    builder = SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.databricks.delta.schema.autoMerge.enabled", "true") \
        .master("local[*]")
    
    logger.info("---Spark session initialized with Delta Lake support---")

    # This helper function adds the Delta Lake JAR packages to the Spark classpath
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    return spark

# Generic function for data quality checks using sampling
def check_data_quality(df, layer_name, sample_ratio=0.1, limit_sample_size=1000):
    """
    Check data quality using sampling to avoid full dataset scans
    
    Parameters:
    - df: DataFrame to analyze
    - layer_name: Name of the layer (bronze, silver, gold)
    - sample_ratio: Fraction of data to sample (0.0 to 1.0)
    - limit_sample_size: Max sample size
    
    Returns: Dictionary with quality metrics
    """
    logger.info(f"Running data quality checks for {layer_name} layer")
    
    # Take a sample with replacement=False to avoid duplicate rows
    sample_df = df.sample(withReplacement=False, fraction=sample_ratio, seed=42) \
                 .limit(limit_sample_size) \
                 .cache()
    
    # Force action to materialize the sample
    sample_df.count()
    
    # Get column info
    columns = df.columns
    column_count = len(columns)
    
    # Collect null statistics in a single pass
    null_counts_expr = [F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c) for c in columns]
    null_stats = sample_df.select(null_counts_expr).first()
    
    # Calculate sample size (more accurate than estimation)
    sample_size = sample_df.count()
    
    # Calculate null percentages
    null_percentages = {col: (null_stats[idx] / sample_size) * 100 if sample_size > 0 else 0 
                       for idx, col in enumerate(columns)}
    
    # Find columns with high null percentages
    high_null_cols = {col: pct for col, pct in null_percentages.items() if pct > 5}
    
    # Clean up cached data
    sample_df.unpersist()
    
    quality_metrics = {
        "layer": layer_name,
        "columns": columns,
        "sample_size": sample_size,
        "column_count": column_count,
        "null_percentages": null_percentages,
        "high_null_columns": high_null_cols,
        "sample_ratio": sample_ratio,
        "timestamp": datetime.now().isoformat()
    }
    
    # Log key metrics
    logger.info(f"Data Quality Metrics for {layer_name} layer:")
    logger.info(f"  - Column Count: {column_count}")
    logger.info(f"  - Columns: {columns}")
    logger.info(f"  - Sample size: {sample_size}")
    
    if high_null_cols:
        logger.warning(f"  - Columns with high null percentages: {high_null_cols}")
    
    return quality_metrics

# Generic validation function that can be used for any layer
def validate_dataframe(df, validation_rules, sample_ratio=0.1, limit_sample_size=1000):
    """
    Validate a dataframe against a set of rules using sampling
    
    Parameters:
    - df: DataFrame to validate
    - validation_rules: List of dictionaries containing validation rules with:
        - name: Rule name
        - condition: SQL condition as a string
        - description: Description of the rule
    - sample_ratio: Fraction of data to sample
    - limit_sample_size: Max sample size
    
    Returns: Dictionary with validation results
    """
    logger.info(f"Validating dataframe with {len(validation_rules)} rules")
    
    # Sample the dataframe
    sample_df = df.sample(withReplacement=False, fraction=sample_ratio, seed=42) \
                  .limit(limit_sample_size) \
                  .cache()
    
    # Force action to materialize the sample
    sample_df.count()
    
    validation_results = {}
    
    # Run each validation rule
    for rule in validation_rules:
        rule_name = rule.get("name", "unnamed_rule")
        condition = rule.get("condition")
        description = rule.get("description", "")
        
        if not condition:
            logger.warning(f"Skipping rule {rule_name} - no condition provided")
            continue
        
        # Count records failing the condition
        failing_count = sample_df.filter(~F.expr(condition)).count()
        
        # Calculate percentage
        sample_size = sample_df.count()
        failing_percentage = (failing_count / sample_size) * 100 if sample_size > 0 else 0
        
        validation_results[rule_name] = {
            "description": description,
            "failing_count": failing_count,
            "failing_percentage": failing_percentage,
            "passed": failing_count == 0
        }
        
        # Log the result
        status = "PASSED" if failing_count == 0 else "FAILED"
        logger.info(f"Validation rule '{rule_name}' {status} - {failing_count} records ({failing_percentage:.2f}%) failed")
    
    # Clean up cached data
    sample_df.unpersist()
    
    return {
        "validation_results": validation_results,
        "sample_size": sample_size,
        "sample_ratio": sample_ratio,
        "timestamp": datetime.now().isoformat(),
        "all_passed": all(result.get("passed", False) for result in validation_results.values())
    }

# Create checkpoint for tracking
def create_checkpoint(pipeline_id, layer, version=None, metadata=None):
    """
    Create a checkpoint record for tracking pipeline progress
    
    Parameters:
    - pipeline_id: Unique identifier for the pipeline run
    - layer: Layer name (bronze, silver, gold)
    - version: Delta table version
    - metadata: Additional metadata to store
    
    Returns: Dictionary with checkpoint information
    """
    checkpoint = {
        "pipeline_id": pipeline_id,
        "layer": layer,
        "version": version,
        "timestamp": datetime.now().isoformat(),
        "metadata": metadata or {}
    }
    
    logger.info(f"Checkpoint created: {checkpoint}")
    return checkpoint

# Process bronze layer
def process_batch_bronze_layer(spark, source_path, schema, bronze_table_path, pipeline_id, mode='test'):
    """
    Process the bronze layer (raw data ingestion)
    
    Parameters:
    - spark: SparkSession
    - source_path: Path to source CSV file
    - schema: Schema for the source data
    - bronze_table_path: Path to store bronze layer Delta table
    - pipeline_id: Unique identifier for this pipeline run
    - mode: 'test' or 'write' mode
    
    Returns: Tuple of (bronzedf, version)
    """
    logger.info(f"Starting bronze layer processing for {source_path}")
    start_time = time.time()
    
    # Read CSV data with schema
    try:
        raw_df = spark.read.format("csv") \
            .option("header", "true") \
            .option("inferSchema", "false") \
            .schema(schema) \
            .load(source_path)

        # Clean any non-timestamp characters first
        raw_df = raw_df.withColumn(
            "timestamp_cleaned", 
            F.regexp_replace(F.col("timestamp"), "[^0-9\\-: ]", "")
        )

        # Use try_cast to handle invalid timestamps gracefully by returning NULL
        raw_df = raw_df.withColumn(
            "timestamp",
            F.expr("try_cast(timestamp_cleaned as timestamp)")
        )

        # Rename to transaction_date
        raw_df = raw_df.withColumnRenamed('timestamp', 'transaction_timestamp')

        # Drop the intermediate column
        raw_df = raw_df.drop("timestamp_cleaned")
            
        logger.info(f"Successfully read CSV data from {source_path}")
            
    except Exception as e:
        logger.error(f"Failed to read CSV data: {str(e)}")
        raise
        
    # Add metadata columns
    bronzedf = raw_df \
        .withColumn("ingestion_timestamp", F.current_timestamp()) \
        .withColumn("source_file", F.input_file_name()) \
        .withColumn("batch_id", F.lit(pipeline_id))
    
    # Define bronze validation rules
    bronze_validation_rules = [
        {
            "name": "has_transaction_id",
            "condition": "transaction_id IS NOT NULL",
            "description": "Transaction ID must be present"
        },
        {
            "name": "valid_amount",
            "condition": "amount IS NULL OR amount > 0",
            "description": "Amount must be positive if not null"
        },
        {
            "name": "valid_transaction_timestamp",
            "condition": "transaction_timestamp IS NULL OR transaction_timestamp <= current_date()",
            "description": "Transaction Timestamp must not be in the future"
        }
    ]
    
    # Write to bronze layer
    try:
        if mode == 'write':
            # Write data to bronze layer
            bronzedf.write \
                .format("delta") \
                .mode("overwrite") \
                .option("overwriteSchema", "true") \
                .save(bronze_table_path)
            
            logger.info(f"Successfully wrote data to bronze layer at {bronze_table_path}")
            
            # Get current version
            delta_table = DeltaTable.forPath(spark, bronze_table_path)
            current_version = delta_table.history(1).select("version").collect()[0][0]
            
            # Run data quality and validation checks
            quality_metrics = check_data_quality(bronzedf, "bronze", sample_ratio=0.2)
            validation_results = validate_dataframe(bronzedf, bronze_validation_rules, sample_ratio=0.2)
            
            # Create checkpoint
            create_checkpoint(pipeline_id, "bronze", current_version, {
                "quality_metrics": quality_metrics,
                "validation_results": validation_results,
                "source_path": source_path,
                "duration_seconds": time.time() - start_time
            })

        elif mode == 'test':
            logger.warning(f"Bronze layer in Test Mode")
            # Run data quality and validation checks
            quality_metrics = check_data_quality(bronzedf, "bronze", sample_ratio=0.2)
            validation_results = validate_dataframe(bronzedf, bronze_validation_rules, sample_ratio=0.2)

            current_version = 'test'

            # Create checkpoint
            create_checkpoint(pipeline_id, "bronze", current_version, {
                "quality_metrics": quality_metrics,
                "validation_results": validation_results,
                "source_path": source_path,
                "duration_seconds": time.time() - start_time
            })

        else:
            raise ValueError("Mode must == 'test' or 'write'")
        
    except Exception as e:
        logger.error(f"Failed to write to bronze layer: {str(e)}")
        raise
    
    logger.info(f"Bronze layer processing completed in {time.time() - start_time:.2f} seconds")
    return bronzedf, current_version

# Process silver layer
def process_batch_silver_layer(spark, bronze_table_path, silver_table_path, pipeline_id, mode='test', bronze_version=None):
    """
    Process the silver layer (cleansed data)
    
    Parameters:
    - spark: SparkSession
    - bronze_table_path: Path to bronze layer Delta table
    - silver_table_path: Path to store silver layer Delta table
    - pipeline_id: Unique identifier for this pipeline run
    - mode: 'test' or 'write' mode
    - bronze_version: Version of bronze data to use
    
    Returns: Tuple of (silverdf, version)
    """
    logger.info("Starting silver layer processing")
    start_time = time.time()
    
    # If bronze_version is provided, read from that specific version
    if bronze_version:
        try:
            bronzedf = spark.read.format("delta") \
                .option("versionAsOf", bronze_version) \
                .load(bronze_table_path)
            logger.info(f"Successfully read bronze data version {bronze_version}")
        except Exception as e:
            logger.error(f"Failed to read bronze data: {str(e)}")
            raise
    else:
        # Otherwise, read the latest version
        try:
            delta_table = DeltaTable.forPath(spark, bronze_table_path)
            bronze_version = delta_table.history(1).select("version").collect()[0][0]
            bronzedf = spark.read.format("delta") \
                .option("versionAsOf", bronze_version) \
                .load(bronze_table_path)
            logger.info(f"Successfully read bronze data version {bronze_version}")
        except Exception as e:
            logger.error(f"Failed to read bronze data: {str(e)}")
            raise

    # Silver layer transformations
    try:
        # Remove duplicates
        silverdf = bronzedf.dropDuplicates(subset=["transaction_id"])

        # Standardize Data
        silverdf = silverdf \
            .withColumn("amount", F.abs(F.col("amount"))) \
            .withColumn("transaction_type", F.lower(F.col("transaction_type"))) \
            .withColumn("category", F.lower(F.col("category"))) \
            .withColumn("status", F.lower(F.col("status")))

        # Filter Data
        # Address bronze layer data validation check concerns
        silverdf = silverdf.filter(
                                    (F.col('transaction_id').isNotNull()) # transaction id must exist
                                    & (F.col('amount') > 0) # amount must be positive
                                    & ((F.col('transaction_timestamp') <= F.current_date()) # must be <= current date
                                    |(F.col('transaction_timestamp').isNull()))) # or must be Null, no future timestamps
        
        # Split timestamp into date and time
        silverdf = silverdf \
            .withColumn("transaction_date", F.to_date("transaction_timestamp")) \
            .withColumn("transaction_time", F.date_format("transaction_timestamp", "HH:mm:ss"))
        
        # Derive year_month for partitioning
        silverdf = silverdf \
            .withColumn("year_month", F.date_format(F.col("transaction_date"), "yyyy-MM")) \
            .withColumn("processing_timestamp", F.current_timestamp())
        
        # Define silver validation rules
        silver_validation_rules = [
            {
                "name": "valid_transaction_type",
                "condition": "transaction_type IN ('debit', 'credit', 'transfer', 'payment', 'withdrawal', 'deposit') OR transaction_type IS NULL",
                "description": "Transaction type must be one of the valid types"
            },
            {
                "name": "valid_status",
                "condition": "status IN ('completed', 'pending', 'failed', 'cancelled', 'refunded') OR status IS NULL",
                "description": "Status must be one of the valid statuses"
            },
            {
                "name": "valid_currency",
                "condition": "currency IS NULL OR length(currency) = 3",
                "description": "Currency code should be 3 characters if present"
            }
        ]
        
        if mode == 'write':
            # Write to silver layer
            silverdf.write \
                .format("delta") \
                .mode("overwrite") \
                .option("overwriteSchema", "true") \
                .partitionBy("year_month") \
                .save(silver_table_path)
            
            logger.info(f"Successfully wrote data to silver layer at {silver_table_path}")
            
            # Get current version
            delta_table = DeltaTable.forPath(spark, silver_table_path)
            current_version = delta_table.history(1).select("version").collect()[0][0]
            
            # Run data quality and validation checks
            quality_metrics = check_data_quality(silverdf, "silver", sample_ratio=0.2)
            validation_results = validate_dataframe(silverdf, silver_validation_rules, sample_ratio=0.2)
            
            # Create checkpoint
            create_checkpoint(pipeline_id, "silver", current_version, {
                "quality_metrics": quality_metrics,
                "validation_results": validation_results,
                "source_bronze_version": bronze_version,
                "duration_seconds": time.time() - start_time
            })
        elif mode == 'test':
            logger.warning(f"--- Silver layer in Test Mode ---")
            # Run data quality and validation checks
            quality_metrics = check_data_quality(silverdf, "silver", sample_ratio=0.2)
            validation_results = validate_dataframe(silverdf, silver_validation_rules, sample_ratio=0.2)

            current_version = 'test'
            # Create checkpoint
            create_checkpoint(pipeline_id, "silver", current_version, {
                "quality_metrics": quality_metrics,
                "validation_results": validation_results,
                "source_bronze_version": bronze_version,
                "duration_seconds": time.time() - start_time
            })
        else:
            raise ValueError("Mode must == 'test' or 'write'")
        
    except Exception as e:
        logger.error(f"Failed to process silver layer: {str(e)}")
        raise
    
    logger.info(f"Silver layer processing completed in {time.time() - start_time:.2f} seconds")
    return silverdf, current_version

# Process gold layer
def process_batch_gold_layer(spark, silver_table_path, gold_table_path, pipeline_id, mode='test', silver_version=None):
    """
    Process the gold layer (business aggregates)
    
    Parameters:
    - spark: SparkSession
    - silver_table_path: Path to silver layer Delta table
    - gold_table_path: Path to store gold layer Delta tables
    - pipeline_id: Unique identifier for this pipeline run
    - mode: 'test' or 'write' mode
    - silver_version: Version of silver data to use
    
    Returns: Dictionary of gold DataFrames
    """
    logger.info("Starting gold layer processing")
    start_time = time.time()
    
    # If silver version is provided, read from that specific version
    if silver_version:
        try:
            silverdf = spark.read.format("delta") \
                .option("versionAsOf", silver_version) \
                .load(silver_table_path)
            logger.info(f"Successfully read silver data version {silver_version}")
        except Exception as e:
            logger.error(f"Failed to read silver data: {str(e)}")
            raise
    else:
        # Otherwise, read the latest version
        try:
            delta_table = DeltaTable.forPath(spark, silver_table_path)
            silver_version = delta_table.history(1).select("version").collect()[0][0]
            silverdf = spark.read.format("delta") \
                .option("versionAsOf", silver_version) \
                .load(silver_table_path)
            logger.info(f"Successfully read silver data version {silver_version}")
        except Exception as e:
            logger.error(f"Failed to read silver data: {str(e)}")
            raise
    
    gold_dfs = {}
    
    try:
        # Gold aggregation 1: Daily summary by category
        daily_category = silverdf \
            .groupBy("transaction_date", "category") \
            .agg(
                F.count("transaction_id").alias("transaction_count"),
                F.sum("amount").alias("total_amount"),
                F.avg("amount").alias("avg_amount"),
                F.min("amount").alias("min_amount"),
                F.max("amount").alias("max_amount"),
                F.countDistinct("customer_id").alias("unique_customers")
            ) \
            .withColumn("processing_timestamp", F.current_timestamp())
        
        gold_dfs["daily_category"] = daily_category
        
        # Gold aggregation 2: Customer summary
        customer_summary = silverdf \
            .groupBy("customer_id") \
            .agg(
                F.count("transaction_id").alias("transaction_count"),
                F.sum("amount").alias("total_amount"),
                F.avg("amount").alias("avg_amount"),
                F.min("transaction_date").alias("first_transaction_date"),
                F.max("transaction_date").alias("last_transaction_date"),
                F.approx_count_distinct("category").alias("category_count")
            ) \
            .withColumn("processing_timestamp", F.current_timestamp()) \
            .withColumn("days_since_last_transaction", 
                        F.datediff(F.current_date(), F.col("last_transaction_date")))
        
        gold_dfs["customer_summary"] = customer_summary
        
        # Gold aggregation 3: Transaction type summary
        transaction_type_summary = silverdf \
            .groupBy("transaction_type") \
            .agg(
                F.count("transaction_id").alias("transaction_count"),
                F.sum("amount").alias("total_amount"),
                F.avg("amount").alias("avg_amount")
            ) \
            .withColumn("processing_timestamp", F.current_timestamp())
        
        gold_dfs["transaction_type_summary"] = transaction_type_summary
        
        # Define gold validation rules
        gold_validation_rules = [
            {
                "name": "positive_transaction_counts",
                "condition": "transaction_count > 0",
                "description": "Transaction counts should be positive"
            },
            {
                "name": "valid_total_amounts",
                "condition": "total_amount >= 0",
                "description": "Total amounts should not be negative"
            }
        ]
        
        if mode == 'write':
            # Write each gold table and validate
            for table_name, df in gold_dfs.items():
                table_path = f"{gold_table_path}/{table_name}"
                
                df.write \
                    .format("delta") \
                    .mode("overwrite") \
                    .option("overwriteSchema", "true") \
                    .save(table_path)
                
                logger.info(f"Successfully wrote gold table {table_name} to {table_path}")
                
                # Get current version
                delta_table = DeltaTable.forPath(spark, table_path)
                current_version = delta_table.history(1).select("version").collect()[0][0]
                
                # Run data quality and validation checks
                quality_metrics = check_data_quality(df, f"gold_{table_name}", sample_ratio=0.5)
                validation_results = validate_dataframe(df, gold_validation_rules, sample_ratio=0.5)
                
                # Create checkpoint
                create_checkpoint(pipeline_id, f"gold_{table_name}", current_version, {
                    "quality_metrics": quality_metrics,
                    "validation_results": validation_results,
                    "source_silver_version": silver_version
                })
        elif mode == 'test':
            logger.warning(f"Gold layer in Test Mode")
            for table_name, df in gold_dfs.items():
                # Run data quality and validation checks
                quality_metrics = check_data_quality(df, f"gold_{table_name}", sample_ratio=0.5)
                validation_results = validate_dataframe(df, gold_validation_rules, sample_ratio=0.5)

                current_version = 'test'
                # Create checkpoint
                create_checkpoint(pipeline_id, f"gold_{table_name}", current_version, {
                    "quality_metrics": quality_metrics,
                    "validation_results": validation_results,
                    "source_silver_version": silver_version
                })
        else:
            raise ValueError("Mode must = test or write")
    except Exception as e:
        logger.error(f"Failed to process gold layer: {str(e)}")
        raise
    
    logger.info(f"Gold layer processing completed in {time.time() - start_time:.2f} seconds")
    return gold_dfs

# Optimize tables
def optimize_table(spark, table_path, zorder_columns: Optional[List[str]] = None):
    """
    Optimize Delta tables for better query performance
    
    Parameters:
    - spark: SparkSession
    - table_path: Path to Delta table
    - zorder_columns: Optional list of columns to Z-ORDER by
    """
    logger.info("Starting table optimization")
    
    try:
        # Get statistics before optimization
        file_stats = spark.sql(f"DESCRIBE DETAIL delta.`{table_path}`").select("numFiles").first()
        file_count_before = file_stats["numFiles"] if file_stats else "unknown"
        
        if zorder_columns:
            zorder_expr = ", ".join(zorder_columns)
            spark.sql(f"OPTIMIZE delta.`{table_path}` ZORDER BY ({zorder_expr})")
        else:
            # Optimize table without Z-ordering
            spark.sql(f"OPTIMIZE delta.`{table_path}`")
        
        # Get statistics after optimization
        file_stats = spark.sql(f"DESCRIBE DETAIL delta.`{table_path}`").select("numFiles").first()
        file_count_after = file_stats["numFiles"] if file_stats else "unknown"
        
        logger.info(f"Optimized {table_path}. Files: before={file_count_before}, after={file_count_after}")
        
    except Exception as e:
        logger.warning(f"Failed to optimize {table_path}: {str(e)}")

# Run the full data engineering pipeline
def run_batch_de_pipeline(spark, source_path, bronze_path, bronze_schema, silver_path, gold_path, pipeline_name='DE_Pipeline'):
    """
    Run a bronze, silver, gold batch ETL pipeline
    
    Parameters:
    - spark: SparkSession
    - source_path: Path to source data
    - bronze_path: Path to store bronze layer
    - bronze_schema: Schema for the source data
    - silver_path: Path to store silver layer
    - gold_path: Path to store gold layer
    - pipeline_name: Name prefix for the pipeline
    
    Returns: Dictionary with pipeline results
    """
    start_time = time.time()
    
    # Generate a unique pipeline ID
    pipeline_id = f"{pipeline_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    pipeline_metrics = {
        "pipeline_id": pipeline_id,
        "start_time": datetime.now().isoformat(),
        "stages": {},
        "status": "running"
    }
    
    try:
        logger.info(f"--Starting data pipeline execution with ID: {pipeline_id}--")
        
        # Process Bronze layer
        bronze_start = time.time()
        bronzedf, bronze_version = process_batch_bronze_layer(spark, source_path, bronze_schema, bronze_path, pipeline_id, 'write')
        bronze_duration = time.time() - bronze_start
        
        pipeline_metrics["stages"]["bronze"] = {
            "duration_seconds": bronze_duration,
            "version": bronze_version,
            "status": "success"
        }
        
        # Optimize Bronze tables
        optimize_start = time.time()
        optimize_table(spark, bronze_path, zorder_columns=None)
        optimize_duration = time.time() - optimize_start
        
        pipeline_metrics["stages"]["bronze_optimize"] = {
            "layer": "bronze",
            "duration_seconds": optimize_duration,
            "status": "success"
        }
        logger.info(f"--Bronze layer successfully optimized in {optimize_duration:.2f} seconds--")
        
        # Process Silver layer
        silver_start = time.time()
        silverdf, silver_version = process_batch_silver_layer(
            spark, bronze_path, silver_path, pipeline_id, 'write', bronze_version
        )
        silver_duration = time.time() - silver_start
        
        pipeline_metrics["stages"]["silver"] = {
            "duration_seconds": silver_duration,
            "version": silver_version,
            "status": "success",
            "source_bronze_version": bronze_version
        }
        
        # Optimize Silver table
        optimize_start = time.time()
        optimize_table(spark, silver_path, zorder_columns=['transaction_date', 'transaction_type'])
        optimize_duration = time.time() - optimize_start
        
        pipeline_metrics["stages"]["silver_optimize"] = {
            "layer": "silver",
            "duration_seconds": optimize_duration,
            "status": "success"
        }
        logger.info(f"--Silver layer successfully optimized in {optimize_duration:.2f} seconds--")
        
        # Process Gold layer
        gold_start = time.time()
        gold_dfs = process_batch_gold_layer(
            spark, silver_path, gold_path, pipeline_id, 'write', silver_version
        )
        gold_duration = time.time() - gold_start
        
        pipeline_metrics["stages"]["gold"] = {
            "duration_seconds": gold_duration,
            "status": "success",
            "source_silver_version": silver_version,
            "tables": list(gold_dfs.keys())
        }
        
        # Optimize Gold tables
        optimize_start = time.time()
        for table in gold_dfs.keys():
            table_path = f"{gold_path}/{table}"
            optimize_table(spark, table_path)
        optimize_duration = time.time() - optimize_start
        
        pipeline_metrics["stages"]["gold_optimize"] = {
            "layer": "gold",
            "duration_seconds": optimize_duration,
            "status": "success"
        }
        logger.info(f"--Gold layer successfully optimized in {optimize_duration:.2f} seconds--")
        
        # Record overall pipeline metrics
        total_duration = time.time() - start_time
        pipeline_metrics["end_time"] = datetime.now().isoformat()
        pipeline_metrics["total_duration_seconds"] = total_duration
        pipeline_metrics["status"] = "success"
        
        logger.info(f"---Full pipeline execution completed successfully in {total_duration:.2f} seconds---")
        
        return {
            "status": "success",
            "pipeline_id": pipeline_id,
            "bronze_version": bronze_version,
            "silver_version": silver_version,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": total_duration,
            "metrics": pipeline_metrics
        }
        
    except Exception as e:
        end_time = time.time()
        error_message = str(e)
        
        pipeline_metrics["end_time"] = datetime.now().isoformat()
        pipeline_metrics["total_duration_seconds"] = end_time - start_time
        pipeline_metrics["status"] = "failed"
        pipeline_metrics["error"] = error_message
        
        logger.error(f"Pipeline failed after {end_time - start_time:.2f} seconds: {error_message}")
        
        return {
            "status": "failed",
            "pipeline_id": pipeline_id,
            "error": error_message,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": end_time - start_time,
            "metrics": pipeline_metrics
        }
    finally:
        # Stop the Spark session
        spark.stop()
        logger.info("Spark session stopped")