import mlflow
import pandas as pd
from lightgbm import LGBMRegressor
from loguru import logger
from mlflow import MlflowClient
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from lightgbm import callback as lgb_callback

import lightgbm as lgb

from house_price.config import ProjectConfig, Tags

"""
infer_signature (from mlflow.models) → Captures input-output schema for model tracking.
"""

"""
num_features → List of numerical feature names.
cat_features → List of categorical feature names.
target → The column to predict.
parameters → Hyperparameters for LightGBM.
catalog_name, schema_name → Database schema names for Databricks tables.
"""


class BasicModel:
    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession):
        """
        Initialize the model with project configuration.
        """
        self.config = config
        self.spark = spark

        # Extract settings from the config
        self.num_features = self.config.num_features
        self.cat_features = self.config.cat_features
        self.target = self.config.target
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name
        self.experiment_name = self.config.experiment_name_basic
        self.model_name = f"{self.catalog_name}.{self.schema_name}.house_prices_model_basic"
        self.tags = tags.dict()
        self.run_id = None

    def load_data(self):
        """
        Load training and testing data from Delta tables.
        Splits data into:
        Features (X_train, X_test)
        Target (y_train, y_test)
        """
        logger.info("🔄 Loading data from Databricks tables...")
        self.train_set_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set")
        self.train_set = self.train_set_spark.toPandas()
        self.test_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_set").toPandas()
        self.data_version = "0"  # describe history -> retrieve

        self.X_train = self.train_set[self.num_features + self.cat_features]
        self.y_train = self.train_set[self.target]
        self.X_test = self.test_set[self.num_features + self.cat_features]
        self.y_test = self.test_set[self.target]
        logger.info("✅ Data successfully loaded.")

    def prepare_features(self):
        """
        Encodes categorical features with OneHotEncoder (ignores unseen categories).
        Passes numerical features as-is (remainder='passthrough').
        Defines a pipeline combining:
            Features processing
            LightGBM regression model
        """
        logger.info("🔄 Defining preprocessing pipeline...")
        self.preprocessor = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_features)], remainder="passthrough"
        )
        
        # Create a custom LGBMRegressor without fitting in the pipeline
        self.lgbm_model = LGBMRegressor(**self.parameters)
        
        # Create a pipeline with just the preprocessor
        self.pipeline = Pipeline(steps=[("preprocessor", self.preprocessor)])
        logger.info("✅ Preprocessing pipeline defined.")


    def log_evaluation(self, env):
        """
        Custom callback to log metrics at each epoch.
        This function is called by LightGBM during training.
        """
        if env.iteration % 1 == 0:  # Log every epoch
            # Access evaluation results safely
            try:
                # The format can vary depending on LightGBM version and configuration
                for i in range(len(env.evaluation_result_list)):
                    item = env.evaluation_result_list[i]
                    
                    # Handle different result formats
                    if isinstance(item, tuple) and len(item) == 2:
                        # Format: (name, value)
                        metric_name, metric_value = item
                        
                        # Try to split the metric name if it contains a colon
                        try:
                            dataset_name, metric = metric_name.split(":")
                        except ValueError:
                            # If splitting fails, use the full name
                            dataset_name = "dataset"
                            metric = metric_name
                            
                    elif isinstance(item, list) and len(item) >= 3:
                        # Format: [data_name, metric_name, metric_value, ...]
                        dataset_name = item[0]
                        metric = item[1]
                        metric_value = item[2]
                    else:
                        # Unknown format, use generic names
                        dataset_name = "dataset"
                        metric = f"metric_{i}"
                        metric_value = float(item)
                    
                    # Log the metric
                    metric_key = f"epoch_{dataset_name}_{metric}"
                    mlflow.log_metric(metric_key, metric_value, step=env.iteration)
                    logger.debug(f"Epoch {env.iteration}: {dataset_name} {metric} = {metric_value}")
                    
            except Exception as e:
                # Log any errors but don't stop training
                logger.error(f"Error in logging evaluation: {e}")
                
        return False

    def train(self):
        """
        Train the model with epoch-level logging.
        """
        logger.info("🚀 Starting training...")
        
        # Start MLflow run
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id
            
            # Log parameters
            mlflow.log_param("model_type", "LightGBM with preprocessing")
            mlflow.log_params(self.parameters)
            
            # Preprocess the data
            X_train_processed = self.pipeline.fit_transform(self.X_train)
            X_test_processed = self.pipeline.transform(self.X_test)
            
            # Set up eval_result dict to store evaluation history
            eval_result = {}
            
            # Train the model with evaluation set
            self.lgbm_model.fit(
                X_train_processed, 
                self.y_train,
                eval_set=[(X_train_processed, self.y_train), (X_test_processed, self.y_test)],
                eval_names=['train', 'validation'],
                eval_metric=['l1', 'l2', 'rmse'],
                callbacks=[lgb.callback.record_evaluation(eval_result), self.log_evaluation],  
                #eval_result=eval_result  # Store evaluation history
            )
            
            # Log the evaluation history at each epoch
            for dataset in eval_result:
                for metric in eval_result[dataset]:
                    for epoch, value in enumerate(eval_result[dataset][metric]):
                        mlflow.log_metric(f"{dataset}_{metric}", value, step=epoch)
            
            # Make predictions
            y_pred = self.lgbm_model.predict(X_test_processed)
            
            # Evaluate metrics
            mse = mean_squared_error(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            logger.info(f"📊 Mean Squared Error: {mse}")
            logger.info(f"📊 Mean Absolute Error: {mae}")
            logger.info(f"📊 R2 Score: {r2}")
            
            # Log final metrics
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2_score", r2)
            
            # Save the complete pipeline for later use
            self.complete_pipeline = Pipeline(
                steps=[("preprocessor", self.preprocessor), ("regressor", self.lgbm_model)]
            )

    def log_model(self):
        """
        Log only the model.
        """
        if not self.run_id:
            logger.error("No active MLflow run. Call train() first.")
            return
        
        with mlflow.start_run(run_id=self.run_id):
            # Create model signature
            signature = infer_signature(model_input=self.X_train, model_output=self.lgbm_model.predict(self.pipeline.transform(self.X_test)))
            
            # Log the dataset
            dataset = mlflow.data.from_spark(
                self.train_set_spark,
                table_name=f"{self.catalog_name}.{self.schema_name}.train_set",
                version=self.data_version,
            )
            mlflow.log_input(dataset, context="training")
            
            # Log the complete model
            mlflow.sklearn.log_model(
                sk_model=self.complete_pipeline, 
                artifact_path="lightgbm-pipeline-model", 
                signature=signature
            )

    def register_model(self):
        """
        Register model in UC
        """
        logger.info("🔄 Registering the model in UC...")
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/lightgbm-pipeline-model",
            name=self.model_name,
            tags=self.tags,
        )
        logger.info(f"✅ Model registered as version {registered_model.version}.")

        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=f"{self.catalog_name}.{self.schema_name}.house_prices_model_basic",
            alias="latest-model",
            version=latest_version,
        )

    def retrieve_current_run_dataset(self):
        """
        Retrieve MLflow run dataset.
        """
        run = mlflow.get_run(self.run_id)
        dataset_info = run.inputs.dataset_inputs[0].dataset
        dataset_source = mlflow.data.get_source(dataset_info)
        logger.info("✅ Dataset source loaded.")
        return dataset_source.load()

    def retrieve_current_run_metadata(self):
        """
        Retrieve MLflow run metadata.
        """
        run = mlflow.get_run(self.run_id)
        metrics = run.data.to_dictionary()["metrics"]
        params = run.data.to_dictionary()["params"]
        logger.info("✅ Dataset metadata loaded.")
        return metrics, params

    def load_latest_model_and_predict(self, input_data: pd.DataFrame):
        """
        Load the latest model from MLflow (alias=latest-model) and make predictions.
        Alias latest is not allowed -> we use latest-model instead as an alternative.

        :param input_data: Pandas DataFrame containing input features for prediction.
        :return: Pandas DataFrame with predictions.
        """
        logger.info("🔄 Loading model from MLflow alias 'production'...")

        model_uri = f"models:/{self.catalog_name}.{self.schema_name}.house_prices_model_basic@latest-model"
        model = mlflow.sklearn.load_model(model_uri)

        logger.info("✅ Model successfully loaded.")

        # Make predictions
        predictions = model.predict(input_data)

        # Return predictions as a DataFrame
        return predictions