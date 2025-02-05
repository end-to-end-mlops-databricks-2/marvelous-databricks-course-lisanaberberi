import logging

import yaml
from pyspark.sql import SparkSession

from house_price.config import ProjectConfig
from house_price.data_processor import DataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

config = ProjectConfig.from_yaml(config_path="../project_config.yml")

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# Load the house prices dataset
spark = SparkSession.builder.getOrCreate()

df = spark.read.csv(
    f"/Volumes/{config.catalog_name}/{config.schema_name}/data/diamonds.csv", header=True, inferSchema=True
).toPandas()

df.show()
# # Force headers to uppercase
# for colname in diamonds_df.columns:
#     if colname == '"table"':
#        new_colname = "TABLE_PCT"
#     else:
#         new_colname = str.upper(colname)
#     diamonds_df = diamonds_df.with_column_renamed(colname, new_colname)

# diamonds_df.show()


# Initialize DataProcessor
