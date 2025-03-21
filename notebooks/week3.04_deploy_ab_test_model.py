# Databricks notebook source

import hashlib

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)
from mlflow.models import infer_signature
from pyspark.sql import SparkSession

from house_price.config import ProjectConfig, Tags
from house_price.models.basic_model import BasicModel

# COMMAND ----------
# Default profile:
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

config = ProjectConfig.from_yaml(config_path="../project_config.yaml")
catalog_name = config.catalog_name
schema_name = config.schema_name
spark = SparkSession.builder.getOrCreate()
tags = Tags(**{"git_sha": "abcd12345", "branch": "week3"})

# COMMAND ----------
# Train & register model A with the config path
basic_model_a = BasicModel(config=config, tags=tags, spark=spark)
basic_model_a.paramaters = config.parameters_a
basic_model_a.model_name = f"{catalog_name}.{schema_name}.house_prices_model_basic_A"
basic_model_a.load_data()
basic_model_a.prepare_features()
basic_model_a.train()
basic_model_a.log_model()
basic_model_a.register_model()
model_A = mlflow.sklearn.load_model(f"models:/{basic_model_a.model_name}@latest-model")

# COMMAND ----------
# Train & register model B with the config path
basic_model_b = BasicModel(config=config, tags=tags, spark=spark)
basic_model_b.paramaters = config.parameters_b
basic_model_b.model_name = f"{catalog_name}.{schema_name}.house_prices_model_basic_B"
basic_model_b.load_data()
basic_model_b.prepare_features()
basic_model_b.train()
basic_model_b.log_model()
basic_model_b.register_model()
model_B = mlflow.sklearn.load_model(f"models:/{basic_model_b.model_name}@latest-model")


# COMMAND ----------
class HousePriceModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, models):
        self.models = models
        self.model_a = models[0]
        self.model_b = models[1]

    def predict(self, context, model_input):
        house_id = str(model_input["Id"].values[0])
        hashed_id = hashlib.md5(house_id.encode(encoding="UTF-8")).hexdigest()
        # convert a hexadecimal (base-16) string into an integer
        if int(hashed_id, 16) % 2:
            predictions = self.model_a.predict(model_input.drop(["Id"], axis=1))
            return {"Prediction": predictions[0], "model": "Model A"}
        else:
            predictions = self.model_b.predict(model_input.drop(["Id"], axis=1))
            return {"Prediction": predictions[0], "model": "Model B"}


# COMMAND ----------
train_set_spark = spark.table(f"{catalog_name}.{schema_name}.train_set")
train_set = train_set_spark.toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()
X_train = train_set[config.num_features + config.cat_features + ["Id"]]
X_test = test_set[config.num_features + config.cat_features + ["Id"]]


# COMMAND ----------
models = [model_A, model_B]
wrapped_model = HousePriceModelWrapper(models)  # we pass the loaded models to the wrapper
example_input = X_test.iloc[0:1]  # Select the first row for prediction as example
example_prediction = wrapped_model.predict(context=None, model_input=example_input)
print("Example Prediction:", example_prediction)

# COMMAND ----------
mlflow.set_experiment(experiment_name="/Shared/house-prices-ab-testing-1")
model_name = f"{catalog_name}.{schema_name}.house_prices_model_pyfunc_ab_test-1"

with mlflow.start_run() as run:
    run_id = run.info.run_id
    signature = infer_signature(model_input=X_train, model_output={"Prediction": 1234.5, "model": "Model B"})
    dataset = mlflow.data.from_spark(train_set_spark, table_name=f"{catalog_name}.{schema_name}.train_set", version="0")
    mlflow.log_input(dataset, context="training")
    mlflow.pyfunc.log_model(
        python_model=wrapped_model, artifact_path="pyfunc-house-price-model-ab", signature=signature
    )
model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/pyfunc-house-price-model-ab", name=model_name, tags=tags.dict()
)

# COMMAND ----------
workspace = WorkspaceClient()
served_entities = [
    ServedEntityInput(
        entity_name=model_name,
        scale_to_zero_enabled=True,
        workload_size="Small",
        entity_version=model_version.version,
    )
]

workspace.serving_endpoints.create(
    name="house-price-model-ab",
    config=EndpointCoreConfigInput(
        served_entities=served_entities,
    ),
)