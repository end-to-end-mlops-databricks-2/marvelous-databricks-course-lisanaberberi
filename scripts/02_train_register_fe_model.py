import mlflow
from pyspark.sql import SparkSession

from house_price.config import ProjectConfig, Tags
from house_price.feature_lookup_model import FeatureLookUpModel

# Configure tracking uri
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

config = ProjectConfig.from_yaml(config_path="../project_config.yml")
spark = SparkSession.builder.getOrCreate()
tags_dict = {"git_sha": "abcd12345", "branch": "week2"}
tags = Tags(**tags_dict)

# Initialize model
fe_model = FeatureLookUpModel(config=config, tags=tags, spark=spark)

# Create feature table
fe_model.create_feature_table()

# Define house age feature function
fe_model.define_feature_function()

# Load data
fe_model.load_data()

# Perform feature engineering
fe_model.feature_engineering()

# Train the model
fe_model.train()

# Register the model
fe_model.register_model()