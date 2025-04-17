import time

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy,
)
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from loguru import logger


class FeatureLookupServing:
    def __init__(self, model_name: str, endpoint_name: str, feature_table_name: str):
        """
        Initializes the Feature Lookup Serving Manager.
        """
        self.workspace = WorkspaceClient()
        self.feature_table_name = feature_table_name
        self.online_table_name = f"{self.feature_table_name}_online"
        self.model_name = model_name
        self.endpoint_name = endpoint_name

    def create_online_table(self):
        """
        Creates an online table for house features.
        """
        spec = OnlineTableSpec(
            primary_key_columns=["Id"],
            source_table_full_name=self.feature_table_name,
            run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({"triggered": "true"}),
            perform_full_copy=False,
        )
        self.workspace.online_tables.create(name=self.online_table_name, spec=spec)

    def get_latest_model_version(self):
        client = mlflow.MlflowClient()
        latest_version = client.get_model_version_by_alias(self.model_name, alias="latest-model").version
        print(f"Latest model version: {latest_version}")
        return latest_version

    def deploy_or_update_serving_endpoint(
        self,
        version: str = "latest",
        workload_size: str = "Small",
        scale_to_zero: bool = True,
    ):
        """
        Deploys the model serving endpoint in Databricks.
        :param version: str. Version of the model to deploy
        :param workload_seze: str. Workload size (number of concurrent requests). Default is Small = 4 concurrent requests
        :param scale_to_zero: bool. If True, endpoint scales to 0 when unused.
        """
        endpoint_exists = any(item.name == self.endpoint_name for item in self.workspace.serving_endpoints.list())
        if version == "latest":
            entity_version = self.get_latest_model_version()
        else:
            entity_version = version

        served_entities = [
            ServedEntityInput(
                entity_name=self.model_name,
                scale_to_zero_enabled=scale_to_zero,
                workload_size=workload_size,
                entity_version=entity_version,
            )
        ]

        if not endpoint_exists:
            self.workspace.serving_endpoints.create(  # create_and_wait
                name=self.endpoint_name,
                config=EndpointCoreConfigInput(
                    served_entities=served_entities,
                ),
            )
        else:
            self.workspace.serving_endpoints.update_config(  # update_config_and_wait
                name=self.endpoint_name,
                served_entities=served_entities,
            )

    def update_online_table(self, config):
        """
        Triggers a Databricks pipeline update and monitors its state.
        """

        update_response = self.workspace.pipelines.start_update(pipeline_id=config.pipeline_id, full_refresh=False)

        while True:
            update_info = self.workspace.pipelines.get_update(
                pipeline_id=config.pipeline_id, update_id=update_response.update_id
            )
            state = update_info.update.state.value

            if state == "COMPLETED":
                logger.info("Pipeline update completed successfully.")
                break
            elif state in ["FAILED", "CANCELED"]:
                logger.error("Pipeline update failed.")
                raise SystemError("Online table failed to update.")
            elif state == "WAITING_FOR_RESOURCES":
                logger.warning("Pipeline is waiting for resources.")
            else:
                logger.info(f"Pipeline is in {state} state.")

            time.sleep(30)