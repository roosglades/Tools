import json
import time
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql import Window

from delta.tables import DeltaTable
from delta import configure_spark_with_delta_pip

# Configure logging
# Sets up basic logging configuration for the pipeline
logger = logging.getLogger('de_pipeline_tools')

if not logger.handlers:
    # Configure just this logger, not the root logger
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    # Prevent propagation to the root logger
    logger.propagate = False

# Initialize Spark session with Delta Lake support
def initialize_local_spark_delta_lake(app_name="DE Pipeline"):
    """
    Initialize a Spark session with Delta Lake support for local execution
    
    Parameters:
    - app_name: Name of the Spark application
    
    Returns:
    - SparkSession: Configured Spark session with Delta Lake support
    """
    # Build a Spark session with Delta Lake extensions and local configuration
    builder = (SparkSession.builder
        .appName(app_name)
        .config("spark.sql.extensions",
                    "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog",
                    "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.databricks.delta.schema.autoMerge.enabled", "true")
        .config("spark.sql.catalogImplementation", "hive")
        .config("spark.sql.warehouse.dir", "C:/hive-warehouse")
        .enableHiveSupport()
        .master("local[*]")  # Uses all available cores on local machine
    )

    logger.info("---Spark session initialized with Delta Lake support---")

    # This helper function adds the Delta Lake JAR packages to the Spark classpath
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    return spark

# Generic function for data quality checks using sampling
def check_data_quality(df, layer_name, sample_ratio=0.05, limit_sample_size=1000):
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
    logger.info("Running data quality checks for %s layer", layer_name)

    # Get approximate row count with timeout and confidence interval
    approx_row_count = df.rdd.countApprox(timeout=100, confidence=0.95)

    # Format schema for pretty logging
    schema_str = df._jdf.schema().treeString()

    # Get column info
    columns = df.columns
    column_count = len(columns)

    # Get shape
    shape = f"[{column_count},{approx_row_count}] (approx. row count)"

    # Take a sample with replacement=False to avoid duplicate rows
    # Cache the sample to avoid recomputation
    sample_df = df.sample(withReplacement=False, fraction=sample_ratio, seed=42) \
                 .limit(limit_sample_size).cache()

    # Get sample size
    sample_size = sample_df.count()

    # Collect null statistics in a single pass using list comprehension
    null_counts_expr = [F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c) for c in columns]
    null_stats = sample_df.select(null_counts_expr).first()

    # Calculate null percentages for each column
    null_percentages = {col: (null_stats[idx] / sample_size) * 100 if sample_size > 0 else 0
                       for idx, col in enumerate(columns)}

    # Find columns with high null percentages (>5%)
    high_null_cols = {col: pct for col, pct in null_percentages.items() if pct > 5}

    # Clean up cached data to free memory
    sample_df.unpersist()

    # Build metrics dictionary with all collected information
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
    logger.info("Data Quality Metrics for %s layer:", layer_name)
    logger.info("  - Shape: %s", shape)
    logger.info("  - Schema: \n%s", schema_str)
    logger.info("  - Sample size: %d", sample_size)

    # Log warning for columns with high null percentages
    if high_null_cols:
        logger.warning("  - Columns with high null percentages: %s", high_null_cols)

    return quality_metrics

# Generic validation function that can be used for any layer
def validate_dataframe(df, validation_rules, sample_ratio=0.05, limit_sample_size=1000):
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
    logger.info("Validating dataframe with %d rules", len(validation_rules))

    # Sample the dataframe and cache it to avoid recomputation
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

        # Skip rules without conditions
        if not condition:
            logger.warning("Skipping rule %s - no condition provided", rule_name)
            continue

        # Count records failing the condition (using negation of the condition)
        failing_count = sample_df.filter(~F.expr(condition)).count()

        # Calculate percentage of failing records
        failing_percentage = (failing_count / sample_size) * 100 if sample_size > 0 else 0

        # Store the validation results
        validation_results[rule_name] = {
            "description": description,
            "failing_count": failing_count,
            "failing_percentage": failing_percentage,
            "passed": failing_count == 0
        }

        # Log the result
        status = "PASSED" if failing_count == 0 else "FAILED"
        logger.info("Validation rule '%s' %s - %d records (%.2f%%) failed",
                   rule_name, status, failing_count, failing_percentage)

    # Clean up cached data to free memory
    sample_df.unpersist()

    # Return comprehensive validation report
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
    # Build the checkpoint dictionary with all relevant information
    checkpoint = {
        "pipeline_id": pipeline_id,
        "layer": layer,
        "version": version,
        "timestamp": datetime.now().isoformat(),
        "metadata": metadata or {}  # Use empty dict if metadata is None
    }

    # Log the checkpoint for monitoring
    logger.info("Checkpoint created:\n%s", json.dumps(checkpoint, indent=2))
    return checkpoint

# Process bronze layer
def process_batch_bronze_layer(spark, source_format, source_path, schema,
                           bronze_table, bronze_transform=None, validation_rules=None,
                           pipeline_id='test', mode='test', bronze_writer=None):
    """
    Process the bronze layer (raw data ingestion)
    """
    logger.info("Starting bronze layer processing")
    start_time = time.time()

    try:
        # Read the source data with specified schema and options
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

        # Log successful data read
        if isinstance(source_path, list):
            paths_str = "\n  - " + "\n  - ".join(source_path)
            logger.info("Successfully read %s data from: %s", source_format.upper(), paths_str)
        else:
            logger.info("Successfully read %s data from: \n%s", source_format.upper(), source_path)

    except Exception as e:
        logger.error("Failed to read %s data: %s", source_format, str(e))
        raise

    try:
        if mode == 'write':
            logger.info("Writing to bronze table: %s", bronze_table)

            try:
                pre_versions = spark.sql(f"DESCRIBE HISTORY {bronze_table}") \
                                   .select("version") \
                                   .orderBy("version", ascending=False) \
                                   .limit(1) \
                                   .collect()
                pre_version = pre_versions[0]["version"] if pre_versions else -1
            except Exception as e:
                logger.warning(f"Table {bronze_table} not found")
                pre_version = -1

            # Pass transform function to writer
            bronze_writer(bronzedf, bronze_table, bronze_transform)

            # Get operations after pre_version
            post_metrics = (
                spark.sql(f"DESCRIBE HISTORY {bronze_table}")
                .filter(f"version > {pre_version}")
                .orderBy("timestamp")
                .select("operationMetrics")
                .collect()
            )
            metrics = [row["operationMetrics"] for row in post_metrics]

            if metrics:
                logger.info("Successfully wrote to bronze table: %s", bronze_table)
                logger.info("Write Metrics: \n%s", json.dumps(metrics, indent=2))
            else:
                logger.info("Nothing written to bronze table: %s", bronze_table)

            delta_table = DeltaTable.forName(spark, bronze_table)
            current_version = delta_table.history(1).select("version").collect()[0][0]

            # For validation, apply transform if needed
            if validation_rules:
                if bronze_transform:
                    validation_df = bronze_transform(bronzedf)
                else:
                    validation_df = bronzedf
                    
                quality_metrics = check_data_quality(validation_df, "bronze", sample_ratio=0.05)
                validation_results = validate_dataframe(validation_df, validation_rules, sample_ratio=0.05)
            else:
                quality_metrics = None
                validation_results = None

            create_checkpoint(pipeline_id, "bronze", current_version, {
                "quality_metrics": quality_metrics,
                "validation_results": validation_results,
                "source_path": source_path,
                "duration_seconds": time.time() - start_time
            })

        elif mode == 'test':
            logger.warning("--- Bronze layer in Test Mode ---")

            if bronze_transform:
                test_df = bronze_transform(bronzedf)
                logger.info("Transformation function applied in test mode")
            else:
                test_df = bronzedf
                logger.info("No transformation function defined")

            if validation_rules:
                quality_metrics = check_data_quality(test_df, "bronze", sample_ratio=0.05)
                validation_results = validate_dataframe(test_df, validation_rules, sample_ratio=0.05)
            else:
                quality_metrics = None
                validation_results = None

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
        logger.error("Failed to write to bronze layer: %s", str(e))
        raise

    logger.info("Bronze layer processing completed in %.2f seconds", time.time() - start_time)
    return bronzedf, current_version

# Process silver layer
def process_batch_silver_layer(spark, bronze_table, bronze_version=None,
                           silver_table=None, silver_transform=None, validation_rules=None,
                           pipeline_id='test', mode='test', silver_writer=None):
    """
    Process the silver layer (cleansed data)
    """
    logger.info("Starting silver layer processing")
    start_time = time.time()

    try:
        if bronze_version:
            bronzedf = (spark.read.format("delta")
                .option("versionAsOf", bronze_version)
                .table(bronze_table)
            )
            logger.info("Successfully read bronze data version %s", bronze_version)
        else:
            delta_table = DeltaTable.forName(spark, bronze_table)
            bronze_version = delta_table.history(1).select("version").collect()[0][0]
            bronzedf = (spark.read.format("delta")
                .option("versionAsOf", bronze_version)
                .table(bronze_table)
            )
            logger.info("Successfully read bronze data version %s", bronze_version)
    except Exception as e:
        logger.error("Failed to read bronze data: %s", str(e))
        raise

    try:
        if mode == 'write':
            logger.info("Writing to silver table: %s", silver_table)

            try:
                pre_versions = spark.sql(f"DESCRIBE HISTORY {silver_table}") \
                                   .select("version") \
                                   .orderBy("version", ascending=False) \
                                   .limit(1) \
                                   .collect()
                pre_version = pre_versions[0]["version"] if pre_versions else -1
            except Exception as e:
                logger.warning(f"Table {silver_table} not found")
                pre_version = -1

            # Pass transform function to writer
            silver_writer(bronzedf, silver_table, silver_transform)

            post_metrics = (
                spark.sql(f"DESCRIBE HISTORY {silver_table}")
                .filter(f"version > {pre_version}")
                .orderBy("timestamp")
                .select("operationMetrics")
                .collect()
            )
            metrics = [row["operationMetrics"] for row in post_metrics]

            if metrics:
                logger.info("Successfully wrote to silver table: %s", silver_table)
                logger.info("Write Metrics: \n%s", json.dumps(metrics, indent=2))
            else:
                logger.info("Nothing written to silver table: %s", silver_table)

            delta_table = DeltaTable.forName(spark, silver_table)
            current_version = delta_table.history(1).select("version").collect()[0][0]

            if validation_rules:
                if silver_transform:
                    validation_df = silver_transform(bronzedf)
                else:
                    validation_df = bronzedf
                
                quality_metrics = check_data_quality(validation_df, "silver", sample_ratio=0.05)
                validation_results = validate_dataframe(validation_df, validation_rules, sample_ratio=0.05)
            else:
                quality_metrics = None
                validation_results = None

            create_checkpoint(pipeline_id, "silver", current_version, {
                "quality_metrics": quality_metrics,
                "validation_results": validation_results,
                "source_bronze_version": bronze_version,
                "duration_seconds": time.time() - start_time
            })

        elif mode == 'test':
            logger.warning("--- Silver layer in Test Mode ---")

            if silver_transform:
                test_df = silver_transform(bronzedf)
                logger.info("Transformation function applied in test mode")
            else:
                test_df = bronzedf
                logger.info("No transformation function defined")

            if validation_rules:
                quality_metrics = check_data_quality(test_df, "silver", sample_ratio=0.05)
                validation_results = validate_dataframe(test_df, validation_rules, sample_ratio=0.05)
            else:
                quality_metrics = None
                validation_results = None

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
        logger.error("Failed to process silver layer: %s", str(e))
        raise

    logger.info("Silver layer processing completed in %.2f seconds", time.time() - start_time)
    return bronzedf, current_version

# Process gold layer
def process_batch_gold_layer(spark, silver_table, silver_version=None,
                          gold_table=None, gold_transform=None, validation_rules=None,
                          pipeline_id='test', mode='test', gold_writer=None):
    """
    Process the gold layer (business aggregates)
    """
    logger.info("Starting gold layer processing")
    start_time = time.time()

    try:
        if silver_version:
            silverdf = (spark.read.format("delta")
                .option("versionAsOf", silver_version)
                .table(silver_table)
            )
            logger.info("Successfully read silver data version %s", silver_version)
        else:
            delta_table = DeltaTable.forName(spark, silver_table)
            silver_version = delta_table.history(1).select("version").collect()[0][0]
            silverdf = (spark.read.format("delta")
                .option("versionAsOf", silver_version)
                .table(silver_table)
            )
            logger.info("Successfully read silver data version %s", silver_version)
    except Exception as e:
        logger.error("Failed to read silver data: %s", str(e))
        raise

    try:
        # Transform stays in the layer processing function for gold layer
        # since it typically creates multiple tables
        gold_dfs = gold_transform(silverdf)
        logger.info("Transformation function applied")
    except Exception as e:
        logger.error("Failed to apply transformation function: %s", str(e))
        raise

    try:
        if mode == 'write':
            for table_name, df in gold_dfs.items():
                full_table_name = f"{gold_table}_{table_name}"
                logger.info("Writing to gold table: %s", full_table_name)

                try:
                    pre_versions = spark.sql(f"DESCRIBE HISTORY {full_table_name}") \
                                     .select("version") \
                                     .orderBy("version", ascending=False) \
                                     .limit(1) \
                                     .collect()
                    pre_version = pre_versions[0]["version"] if pre_versions else -1
                except Exception as e:
                    logger.warning(f"Table {full_table_name} not found")
                    pre_version = -1

                # Write directly since transform already applied
                gold_writer(df, full_table_name)

                post_metrics = (
                    spark.sql(f"DESCRIBE HISTORY {full_table_name}")
                    .filter(f"version > {pre_version}")
                    .orderBy("timestamp")
                    .select("operationMetrics")
                    .collect()
                )
                metrics = [row["operationMetrics"] for row in post_metrics]

                if metrics:
                    logger.info("Successfully wrote to gold table: %s", full_table_name)
                    logger.info("Write Metrics: \n%s", json.dumps(metrics, indent=2))
                else:
                    logger.info("Nothing written to gold table: %s", full_table_name)

                delta_table = DeltaTable.forName(spark, full_table_name)
                current_version = delta_table.history(1).select("version").collect()[0][0]

                if validation_rules:
                    quality_metrics = check_data_quality(df, f"{full_table_name}", sample_ratio=0.05)
                    validation_results = validate_dataframe(df, validation_rules, sample_ratio=0.05)
                else:
                    quality_metrics = None
                    validation_results = None

                create_checkpoint(pipeline_id, f"gold_{table_name}", current_version, {
                    "quality_metrics": quality_metrics,
                    "validation_results": validation_results,
                    "source_silver_version": silver_version
                })

        elif mode == 'test':
            logger.warning("Gold layer in Test Mode")
            for table_name, df in gold_dfs.items():
                full_table_name = f"{gold_table}_{table_name}"

                if validation_rules:
                    quality_metrics = check_data_quality(df, f"{full_table_name}", sample_ratio=0.05)
                    validation_results = validate_dataframe(df, validation_rules, sample_ratio=0.05)
                else:
                    quality_metrics = None
                    validation_results = None

                current_version = 'test'
                create_checkpoint(pipeline_id, f"gold_{table_name}", current_version, {
                    "quality_metrics": quality_metrics,
                    "validation_results": validation_results,
                    "source_silver_version": silver_version
                })
        else:
            raise ValueError("Mode must = 'test' or 'write'")

    except Exception as e:
        logger.error("Failed to process gold layer: %s", str(e))
        raise

    logger.info("Gold layer processing completed in %.2f seconds", time.time() - start_time)
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

        # Apply Z-ORDER optimization if columns specified
        if zorder_columns:
            zorder_expr = ", ".join(zorder_columns)
            spark.sql(f"OPTIMIZE {table} ZORDER BY ({zorder_expr})")
        else:
            # Optimize table without Z-ordering
            spark.sql(f"OPTIMIZE {table}")

        # Get statistics after optimization
        file_stats = spark.sql(f"DESCRIBE DETAIL {table}").select("numFiles").first()
        file_count_after = file_stats["numFiles"] if file_stats else "unknown"

        # Log results of optimization
        logger.info("Optimized %s. Files: before=%s, after=%s",
                   table, file_count_before, file_count_after)

    except RuntimeError as e:
        logger.warning("Failed to optimize %s: %s", table, str(e))

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
    - gold_transform:   Function to create gold DataFrames (takes DataFrame, returns Dict)
    - bronze_validation_rules: Validation rules for bronze layer
    - silver_validation_rules: Validation rules for silver layer
    - gold_validation_rules: Validation rules for gold layer
    - pipeline_name: Name prefix for the pipeline
    
    Returns:
    - Dict: Dictionary with pipeline results and metrics
    """
    start_time = time.time()

    # Generate a unique pipeline ID with timestamp
    pipeline_id = f"{pipeline_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Initialize pipeline metrics tracking
    pipeline_metrics = {
        "pipeline_id": pipeline_id,
        "start_time": datetime.now().isoformat(),
        "stages": {},
        "status": "running"
    }

    try:
        logger.info("--Starting data pipeline execution with ID: %s--", pipeline_id)

        # Process Bronze layer
        bronze_start = time.time()
        bronzedf, bronze_version = process_batch_bronze_layer(spark, source_format, source_path,
                                                            bronze_schema, bronze_table,
                                                            bronze_transform, bronze_validation_rules,
                                                            pipeline_id, 'write', bronze_writer)
        bronze_duration = time.time() - bronze_start

        # Record bronze metrics
        pipeline_metrics["stages"]["bronze"] = {
            "duration_seconds": bronze_duration,
            "version": bronze_version,
            "status": "success"
        }

        # Optimize Bronze tables
        optimize_start = time.time()
        optimize_table(spark, bronze_table, zorder_columns=None)
        optimize_duration = time.time() - optimize_start

        # Record bronze optimization metrics
        pipeline_metrics["stages"]["bronze_optimize"] = {
            "layer": "bronze",
            "duration_seconds": optimize_duration,
            "status": "success"
        }
        logger.info("--Bronze layer successfully optimized in %.2f seconds--", optimize_duration)

        # Process Silver layer
        silver_start = time.time()
        silverdf, silver_version = process_batch_silver_layer(spark, bronze_table, bronze_version,
                                                            silver_table, silver_transform,
                                                            silver_validation_rules, pipeline_id,
                                                            'write', silver_writer)
        silver_duration = time.time() - silver_start

        # Record silver metrics
        pipeline_metrics["stages"]["silver"] = {
            "duration_seconds": silver_duration,
            "version": silver_version,
            "status": "success",
            "source_bronze_version": bronze_version
        }

        # Optimize Silver table with Z-ORDER on common query columns
        optimize_start = time.time()
        optimize_table(spark, silver_table, zorder_columns=['transaction_date', 'transaction_type'])
        optimize_duration = time.time() - optimize_start

        # Record silver optimization metrics
        pipeline_metrics["stages"]["silver_optimize"] = {
            "layer": "silver",
            "duration_seconds": optimize_duration,
            "status": "success"
        }
        logger.info("--Silver layer successfully optimized in %.2f seconds--", optimize_duration)

        # Process Gold layer only if gold_table is specified
        if gold_table:

            # Process Gold layer - creates multiple aggregated tables
            gold_start = time.time()
            gold_dfs = process_batch_gold_layer(spark, silver_table, silver_version,
                                              gold_table, gold_transform, gold_validation_rules,
                                              pipeline_id, 'write', gold_writer)

            gold_duration = time.time() - gold_start

            # Record gold metrics
            pipeline_metrics["stages"]["gold"] = {
                "duration_seconds": gold_duration,
                "status": "success",
                "source_silver_version": silver_version,
                "tables": list(gold_dfs.keys())
            }

            # Optimize all Gold tables
            optimize_start = time.time()
            for table_name in gold_dfs.keys():
                full_table_name = f"{gold_table}_{table_name}"
                optimize_table(spark, full_table_name)
            optimize_duration = time.time() - optimize_start

            # Record gold optimization metrics
            pipeline_metrics["stages"]["gold_optimize"] = {
                "layer": "gold",
                "duration_seconds": optimize_duration,
                "status": "success"
            }
            logger.info("--Gold layer successfully optimized in %.2f seconds--", optimize_duration)

        else:
            logger.info("-- Gold layer not defined --")

        # Record overall pipeline metrics
        total_duration = time.time() - start_time
        pipeline_metrics["end_time"] = datetime.now().isoformat()
        pipeline_metrics["total_duration_seconds"] = total_duration
        pipeline_metrics["status"] = "success"

        logger.info("---Full pipeline execution completed successfully in %.2f seconds---",
                    total_duration)

        # Return comprehensive pipeline results
        return {
            "status": "success",
            "pipeline_id": pipeline_id,
            "bronze_version": bronze_version,
            "silver_version": silver_version,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": total_duration,
            "metrics": pipeline_metrics
        }

    except RuntimeError as e:
        # Handle pipeline failure
        end_time = time.time()
        error_message = str(e)

        # Record error information
        pipeline_metrics["end_time"] = datetime.now().isoformat()
        pipeline_metrics["total_duration_seconds"] = end_time - start_time
        pipeline_metrics["status"] = "failed"
        pipeline_metrics["error"] = error_message

        logger.error("Pipeline failed after %.2f seconds: %s", end_time - start_time, error_message)

        # Return failure information
        return {
            "status": "failed",
            "pipeline_id": pipeline_id,
            "error": error_message,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": end_time - start_time,
            "metrics": pipeline_metrics
        }
    finally:
        # Always stop the Spark session to free resources
        spark.stop()
        logger.info("Spark session stopped")
