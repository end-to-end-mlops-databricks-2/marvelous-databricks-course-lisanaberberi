# Databricks notebook source
#  install the wheel package
#%pip install file:///Volumes/mlops_dev/lisanabe/packages/house_price-0.0.1-py3-none-any.whl

import mlflow
from pyspark.sql import SparkSession

from house_price.config import ProjectConfig, Tags
from house_price.models.basic_model import BasicModel

# COMMAND ----------
# Default profile:
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# Profile called "course"
# mlflow.set_tracking_uri("databricks://course")
# mlflow.set_registry_uri("databricks-uc://course")

config = ProjectConfig.from_yaml(config_path="../project_config.yaml")
spark = SparkSession.builder.getOrCreate()
tags = Tags(**{"git_sha": "abcd12345", "branch": "week2"})

# COMMAND ----------
# Initialize model with the config path
basic_model = BasicModel(config=config, tags=tags, spark=spark)

# COMMAND ----------
basic_model.load_data()
basic_model.prepare_features()

# COMMAND ----------
# Train + log the model (runs everything including MLflow logging)
basic_model.train()
basic_model.log_model()

# COMMAND ----------
run_id = mlflow.search_runs(
    experiment_names=["/Shared/house-prices-basic"], filter_string="tags.branch='week2'"
).run_id[0]

model = mlflow.sklearn.load_model(f"runs:/{run_id}/lightgbm-pipeline-model")

# COMMAND ----------
# Retrieve dataset for the current run
basic_model.retrieve_current_run_dataset()

# COMMAND ----------
# Retrieve metadata for the current run
basic_model.retrieve_current_run_metadata()

# COMMAND ----------
# Register model
basic_model.register_model()

# COMMAND ----------
# Predict on the test set

test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(10)

X_test = test_set.drop(config.target).toPandas()

predictions_df = basic_model.load_latest_model_and_predict(X_test)
# COMMAND ----------