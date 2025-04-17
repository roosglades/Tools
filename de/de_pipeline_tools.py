import json
import time
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from delta.tables import DeltaTable
from delta import configure_spark_with_delta_pip

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Spark session with Delta Lake support
def initialize_local_spark_delta_lake(app_name="DE Pipeline"):
    """
    Initialize a Spark session with Delta Lake support for local execution
    
    Parameters:
    - app_name: Name of the Spark application
    
    Returns:
    - SparkSession: Configured Spark session with Delta Lake support
    """
    builder = (SparkSession.builder
        .appName(app_name)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.databricks.delta.schema.autoMerge.enabled", "true")
        .config("spark.sql.catalogImplementation", "hive")
        .config("spark.sql.warehouse.dir", "C:/hive-warehouse")
        .enableHiveSupport()
        .master("local[*]")
    )
    
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
    
    Returns:
    - Dict: Dictionary with quality metrics including null percentages and column stats
    """
    logger.info(f"Running data quality checks for {layer_name} layer")

    approx_row_count    = df.rdd.countApprox(timeout=100, confidence=0.95)

    # Format schema for pretty logging
    schema_str = df._jdf.schema().treeString()
    
    # Get column info
    columns = df.columns
    column_count = len(columns)

    # Get shape
    shape = f"[{column_count},{approx_row_count}] (approx. row count)"

    # Take a sample with replacement=False to avoid duplicate rows
    sample_df = df.sample(withReplacement=False, fraction=sample_ratio, seed=42) \
                 .limit(limit_sample_size).cache()
    
    # Get sample size
    sample_size = sample_df.count()
    
    # Collect null statistics in a single pass
    null_counts_expr = [F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c) for c in columns]
    null_stats = sample_df.select(null_counts_expr).first()
    
    # Calculate null percentages
    null_percentages = {col: (null_stats[idx] / sample_size) * 100 if sample_size > 0 else 0 
                       for idx, col in enumerate(columns)}
    
    # Find columns with high null percentages
    high_null_cols = {col: pct for col, pct in null_percentages.items() if pct > 5}
    
    # Clean up cached data
    sample_df.unpersist()
    
    quality_metrics = {
        "layer": layer_name,
        "shape": shape,
        "schema": schema_str,
        "sample_size": sample_size,
        "null_percentages": null_percentages,
        "high_null_columns": high_null_cols,
        "sample_ratio": sample_ratio,
        "timestamp": datetime.now().isoformat()
    }
    
    # Log key metrics
    logger.info(f"Data Quality Metrics for {layer_name} layer:")
    logger.info(f"  - Shape: {shape}")
    logger.info(f"  - Schema: \n{schema_str}")
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
    
    Returns:
    - Dict: Dictionary with validation results including pass/fail status for each rule
    """
    logger.info(f"Validating dataframe with {len(validation_rules)} rules")
    
    # Sample the dataframe
    sample_df = df.sample(withReplacement=False, fraction=sample_ratio, seed=42) \
                  .limit(limit_sample_size) \
                  .cache()
    
    # Get sample size
    sample_size = sample_df.count()
    
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
    
    Returns:
    - Dict: Dictionary with checkpoint information
    """
    checkpoint = {
        "pipeline_id": pipeline_id,
        "layer": layer,
        "version": version,
        "timestamp": datetime.now().isoformat(),
        "metadata": metadata or {}
    }
    
    logger.info(f"Checkpoint created:\n{json.dumps(checkpoint, indent=2)}")
    return checkpoint

# Process bronze layer
def process_batch_bronze_layer(spark, source_format, source_path, schema,
                                bronze_table, bronze_transform=None, validation_rules=None,
                                pipeline_id='test', mode='test', bronze_writer=None):
    """
    Process the bronze layer (raw data ingestion)
    
    Parameters:
    - spark: SparkSession
    - source_format: Format of source data (e.g., 'csv', 'json', 'parquet','delta')
    - source_path: Path to source data
    - schema: Schema for the source data
    - bronze_table: Name for the bronze layer Delta table
    - bronze_transform: Function to transform the bronze DataFrame (takes DataFrame, returns DataFrame)
    - validation_rules: List of validation rules to apply
    - pipeline_id: Unique identifier for this pipeline run
    - mode: 'test' or 'write' mode
    
    Returns:
    - Tuple: (DataFrame, version) - Bronze DataFrame and its version
    """
    logger.info("Starting bronze layer processing")
    start_time = time.time()
    
    # Read data with schema
    try:
        bronzedf = (spark.read.format(source_format)
            .option("header", "true")
            .option("inferSchema", "false")
            .schema(schema)
            .load(source_path)
        )

        # Add metadata columns
        bronzedf = (bronzedf
                .withColumn("ingestion_timestamp", F.current_timestamp())
                .withColumn("source_file", F.input_file_name())
                .withColumn("batch_id", F.lit(pipeline_id))
        )

        if isinstance(source_path, list):
            paths_str = "\n  - " + "\n  - ".join(source_path)
            logger.info(f"Successfully read {source_format.upper()} data from: {paths_str}")
        else:
            logger.info(f"Successfully read {source_format.upper()} data from: \n{source_path}")
            
    except Exception as e:
        logger.error(f"Failed to read {source_format} data: {str(e)}")
        raise

    # Transform Bronze (None is normal)
    try:
        if bronze_transform:
                bronzedf = bronze_transform(bronzedf)
                logger.info(f"Transformation function applied")
            
        else:
            logger.info(f"No transformation function defined")
    
    except Exception as e:
        logger.error(f"Failed to apply transformation function")
        raise

    try:
        if mode == 'write':
            # Execute defined bronze writer function if provided
            metrics = bronze_writer(bronzedf, bronze_table)
            
            logger.info(f"Successfully wrote bronze table: {bronze_table}")

            if metrics:
                logger.info(f"Write Metrics: \n{json.dumps(metrics, indent=2)}")
            
            # Get current version
            delta_table = DeltaTable.forName(spark, bronze_table)
            current_version = delta_table.history(1).select("version").collect()[0][0]
            
            # Run data quality and validation checks
            quality_metrics = check_data_quality(bronzedf, "bronze", sample_ratio=0.1)
            if validation_rules:
                validation_results = validate_dataframe(bronzedf, validation_rules, sample_ratio=0.1)
            else:
                validation_results = None
            
            # Create checkpoint
            create_checkpoint(pipeline_id, "bronze", current_version, {
                "quality_metrics": quality_metrics,
                "validation_results": validation_results,
                "source_path": source_path,
                "duration_seconds": time.time() - start_time
            })

        elif mode == 'test':
            logger.warning(f"--- Bronze layer in Test Mode ---")
            # Run data quality and validation checks
            quality_metrics = check_data_quality(bronzedf, "bronze", sample_ratio=0.1)
            if validation_rules:
                validation_results = validate_dataframe(bronzedf, validation_rules, sample_ratio=0.1)
            else:
                validation_results = None

            # Create checkpoint
            current_version = 'test'
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
def process_batch_silver_layer(spark, bronze_table, bronze_version=None, 
                                silver_table=None, silver_transform=None, validation_rules=None,
                                pipeline_id='test', mode='test', silver_writer=None):
    """
    Process the silver layer (cleansed data)
    
    Parameters:
    - spark: SparkSession
    - bronze_table: Name of bronze layer Delta table
    - silver_table: Name for the silver layer Delta table
    - silver_transform: Function to transform the silver DataFrame (takes DataFrame, returns DataFrame)
    - validation_rules: List of validation rules to apply
    - pipeline_id: Unique identifier for this pipeline run
    - mode: 'test' or 'write' mode
    - bronze_version: Version of bronze data to use (None for latest)
    
    Returns:
    - Tuple: (DataFrame, version) - Silver DataFrame and its version
    """
    logger.info("Starting silver layer processing")
    start_time = time.time()
    
    try:
        # If bronze_version is provided, read from that specific version
        if bronze_version:
            bronzedf = (spark.read.format("delta")
                .option("versionAsOf", bronze_version)
                .table(bronze_table)
            )

            logger.info(f"Successfully read bronze data version {bronze_version}")

        # Otherwise, read the latest version
        else:
            delta_table = DeltaTable.forName(spark, bronze_table)
            bronze_version = delta_table.history(1).select("version").collect()[0][0]

            bronzedf = (spark.read.format("delta")
                .option("versionAsOf", bronze_version)
                .table(bronze_table)
            )

            logger.info(f"Successfully read bronze data version {bronze_version}")

    except Exception as e:
        logger.error(f"Failed to read bronze data: {str(e)}")
        raise
    
    # Transform Silver
    try:
        if silver_transform:
                silverdf = silver_transform(bronzedf)
                logger.info(f"Transformation function applied")
            
        else:
            silverdf = bronzedf
            logger.warning(f"No transformation function defined")

    except Exception as e:
        logger.error(f"Failed to apply transformation function")
        raise

    try:        
        if mode == 'write':
            # Write to silver layer
            metrics = silver_writer(silverdf, silver_table)
            
            logger.info(f"Successfully wrote silver table: {silver_table}")

            if metrics:
                logger.info(f"Write Metrics: \n{json.dumps(metrics, indent=2)}")
            
            # Get current version
            delta_table = DeltaTable.forName(spark, silver_table)
            current_version = delta_table.history(1).select("version").collect()[0][0]
            
            # Run data quality and validation checks
            quality_metrics = check_data_quality(silverdf, "silver", sample_ratio=0.1)
            if validation_rules:
                validation_results = validate_dataframe(silverdf, validation_rules, sample_ratio=0.1)
            else:
                validation_results = None
            
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
            quality_metrics = check_data_quality(silverdf, "silver", sample_ratio=0.1)
            if validation_rules:
                validation_results = validate_dataframe(silverdf, validation_rules, sample_ratio=0.1)
            else: 
                validation_results = None

            # Create checkpoint
            current_version = 'test'
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
def process_batch_gold_layer(spark, silver_table, silver_version=None, 
                            gold_table=None, gold_transform=None, validation_rules=None, 
                            pipeline_id='test', mode='test', gold_writer=None):
    """
    Process the gold layer (business aggregates)
    
    Parameters:
    - spark: SparkSession
    - silver_table: Name of silver layer Delta table
    - gold_table: Table prefix to store gold layer tables
    - gold_transform: Function that transforms silver DataFrame into multiple gold DataFrames 
                     (takes DataFrame, returns Dict of DataFrames)
    - validation_rules: List of validation rules to apply
    - pipeline_id: Unique identifier for this pipeline run
    - mode: 'test' or 'write' mode
    - silver_version: Version of silver data to use (None for latest)
    
    Returns:
    - Dict: Dictionary of gold DataFrames with table names as keys
    """
    logger.info("Starting gold layer processing")
    start_time = time.time()
    

    try:
        # If silver version is provided, read from that specific version
        if silver_version:
            silverdf = (spark.read.format("delta")
                .option("versionAsOf", silver_version)
                .table(silver_table)
            )
            logger.info(f"Successfully read silver data version {silver_version}")

        else:
            # Otherwise, read the latest version
            delta_table = DeltaTable.forName(spark, silver_table)
            silver_version = delta_table.history(1).select("version").collect()[0][0]

            silverdf = (spark.read.format("delta")
                .option("versionAsOf", silver_version)
                .table(silver_table)
            )
            logger.info(f"Successfully read silver data version {silver_version}")

    except Exception as e:
        logger.error(f"Failed to read silver data: {str(e)}")
        raise
    
    # Transform(s) Gold
    try:
        gold_dfs = gold_transform(silverdf)
        logger.info(f"Transformation function applied")

    except Exception as e:
        logger.error(f"Failed to apply transformation function")
        raise
    
    
    try:        
        if mode == 'write':
            # Write each gold table and validate
            for table_name, df in gold_dfs.items():
                full_table_name = f"{gold_table}_{table_name}"
                metrics = gold_writer(df, full_table_name)
                
                logger.info(f"Successfully wrote gold table: {full_table_name}")

                if metrics:
                    logger.info(f"Write Metrics: \n{json.dumps(metrics, indent=2)}")
                
                # Get current version
                delta_table = DeltaTable.forName(spark, full_table_name)
                current_version = delta_table.history(1).select("version").collect()[0][0]
                
                # Run data quality and validation checks
                quality_metrics = check_data_quality(df, f"{full_table_name}", sample_ratio=0.2)
                if validation_rules:
                    validation_results = validate_dataframe(df, validation_rules, sample_ratio=0.2)
                else:
                    validation_results = None
                
                # Create checkpoint
                create_checkpoint(pipeline_id, f"gold_{table_name}", current_version, {
                    "quality_metrics": quality_metrics,
                    "validation_results": validation_results,
                    "source_silver_version": silver_version
                })
        elif mode == 'test':
            logger.warning(f"Gold layer in Test Mode")
            for table_name, df in gold_dfs.items():
                full_table_name = f"{gold_table}_{table_name}"
                # Run data quality and validation checks
                quality_metrics = check_data_quality(df, f"{full_table_name}", sample_ratio=0.2)
                if validation_rules:
                    validation_results = validate_dataframe(df, validation_rules, sample_ratio=0.2)
                else:
                    validation_results = None

                # Create checkpoint
                current_version = 'test'
                create_checkpoint(pipeline_id, f"gold_{table_name}", current_version, {
                    "quality_metrics": quality_metrics,
                    "validation_results": validation_results,
                    "source_silver_version": silver_version
                })
        else:
            raise ValueError("Mode must = 'test' or 'write'")

    except Exception as e:
        logger.error(f"Failed to process gold layer: {str(e)}")
        raise
    
    logger.info(f"Gold layer processing completed in {time.time() - start_time:.2f} seconds")

    return gold_dfs

# Optimize tables
def optimize_table(spark, table, zorder_columns: Optional[List[str]] = None):
    """
    Optimize Delta tables for better query performance
    
    Parameters:
    - spark: SparkSession
    - table: Name of Delta table to optimize
    - zorder_columns: Optional list of columns to Z-ORDER by
    """
    logger.info("Starting table optimization")
    
    try:
        # Get statistics before optimization
        file_stats = spark.sql(f"DESCRIBE DETAIL {table}").select("numFiles").first()
        file_count_before = file_stats["numFiles"] if file_stats else "unknown"
        
        if zorder_columns:
            zorder_expr = ", ".join(zorder_columns)
            spark.sql(f"OPTIMIZE {table} ZORDER BY ({zorder_expr})")
        else:
            # Optimize table without Z-ordering
            spark.sql(f"OPTIMIZE {table}")
        
        # Get statistics after optimization
        file_stats = spark.sql(f"DESCRIBE DETAIL {table}").select("numFiles").first()
        file_count_after = file_stats["numFiles"] if file_stats else "unknown"
        
        logger.info(f"Optimized {table}. Files: before={file_count_before}, after={file_count_after}")
        
    except Exception as e:
        logger.warning(f"Failed to optimize {table}: {str(e)}")

# Run the full data engineering pipeline
def run_batch_de_pipeline(spark, source_format, source_path,
                        bronze_schema, bronze_table=None, silver_table=None, gold_table=None, 
                        bronze_transform=None, silver_transform=None, gold_transform=None, 
                        bronze_writer=None, silver_writer=None, gold_writer=None,
                        bronze_validation_rules=None, silver_validation_rules=None, gold_validation_rules=None,
                        pipeline_name='DE_Pipeline'):
    """
    Run a bronze, silver, gold batch ETL pipeline
    
    Parameters:
    - spark: SparkSession
    - source_format: Format of source data (e.g., 'csv', 'json', 'parquet')
    - source_path: Path to source data
    - bronze_schema: Schema for the source data
    - bronze_table: Name for the bronze layer Delta table
    - silver_table: Name for the silver layer Delta table
    - gold_table: Table prefix for gold layer tables
    - bronze_transform: Function to transform bronze DataFrame (takes DataFrame, returns DataFrame)
    - silver_transform: Function to transform silver DataFrame (takes DataFrame, returns DataFrame)
    - gold_transform: Function to create gold DataFrames (takes DataFrame, returns Dict of DataFrames)
    - bronze_validation_rules: Validation rules for bronze layer
    - silver_validation_rules: Validation rules for silver layer
    - gold_validation_rules: Validation rules for gold layer
    - pipeline_name: Name prefix for the pipeline
    
    Returns:
    - Dict: Dictionary with pipeline results and metrics
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
        bronzedf, bronze_version = process_batch_bronze_layer(spark, source_format, source_path, bronze_schema, bronze_table, 
                                                            bronze_transform, bronze_validation_rules, pipeline_id, 
                                                            'write', bronze_writer)
        bronze_duration = time.time() - bronze_start
        
        pipeline_metrics["stages"]["bronze"] = {
            "duration_seconds": bronze_duration,
            "version": bronze_version,
            "status": "success"
        }
        
        # Optimize Bronze tables
        optimize_start = time.time()
        optimize_table(spark, bronze_table, zorder_columns=None)
        optimize_duration = time.time() - optimize_start
        
        pipeline_metrics["stages"]["bronze_optimize"] = {
            "layer": "bronze",
            "duration_seconds": optimize_duration,
            "status": "success"
        }
        logger.info(f"--Bronze layer successfully optimized in {optimize_duration:.2f} seconds--")
        
        # Process Silver layer
        silver_start = time.time()
        silverdf, silver_version = process_batch_silver_layer(spark, bronze_table, bronze_version, 
                                                                silver_table, silver_transform, silver_validation_rules, 
                                                                pipeline_id, 'write', silver_writer)
        silver_duration = time.time() - silver_start
        
        pipeline_metrics["stages"]["silver"] = {
            "duration_seconds": silver_duration,
            "version": silver_version,
            "status": "success",
            "source_bronze_version": bronze_version
        }
        
        # Optimize Silver table
        optimize_start = time.time()
        optimize_table(spark, silver_table, zorder_columns=['transaction_date', 'transaction_type'])
        optimize_duration = time.time() - optimize_start
        
        pipeline_metrics["stages"]["silver_optimize"] = {
            "layer": "silver",
            "duration_seconds": optimize_duration,
            "status": "success"
        }
        logger.info(f"--Silver layer successfully optimized in {optimize_duration:.2f} seconds--")

        if gold_table:
        
            # Process Gold layer
            gold_start = time.time()
            gold_dfs = process_batch_gold_layer(spark, silver_table, silver_version,
                                                gold_table, gold_transform, gold_validation_rules,
                                                pipeline_id, 'write', gold_writer)

            gold_duration = time.time() - gold_start
            
            pipeline_metrics["stages"]["gold"] = {
                "duration_seconds": gold_duration,
                "status": "success",
                "source_silver_version": silver_version,
                "tables": list(gold_dfs.keys())
            }
            
            # Optimize Gold tables
            optimize_start = time.time()
            for table_name in gold_dfs.keys():
                full_table_name = f"{gold_table}_{table_name}"
                optimize_table(spark, full_table_name)
            optimize_duration = time.time() - optimize_start
            
            pipeline_metrics["stages"]["gold_optimize"] = {
                "layer": "gold",
                "duration_seconds": optimize_duration,
                "status": "success"
            }
            logger.info(f"--Gold layer successfully optimized in {optimize_duration:.2f} seconds--")
        
        else:
            logger.info(f"-- Gold layer not defined --")
        
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