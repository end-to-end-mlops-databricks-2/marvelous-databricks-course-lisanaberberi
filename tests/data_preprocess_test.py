import pytest
import pandas as pd
from unittest.mock import MagicMock
from pyspark.sql import SparkSession
from house_price.config import ProjectConfig
from house_price.dataprocessor import DataProcessor

def mock_project_config():
    return ProjectConfig(
        num_features=["LotFrontage", "MasVnrArea"],
        cat_features=["MasVnrType"],
        target="SalePrice",
        catalog_name="house_price_catalog",
        schema_name="house_price_schema"
    )

@pytest.fixture
def sample_dataframe():
    data = {
        "Id": [1, 2, 3],
        "LotFrontage": [80, None, 75],
        "MasVnrType": [None, "BrkFace", "None"],
        "MasVnrArea": [None, 120, 0],
        "GarageYrBlt": [2000, None, 1995],
        "SalePrice": [200000, 150000, 180000]
    }
    return pd.DataFrame(data)

@pytest.fixture
def spark_session():
    return SparkSession.builder.master("local").appName("pytest").getOrCreate()

@pytest.fixture
def data_processor(sample_dataframe, spark_session):
    config = mock_project_config()
    return DataProcessor(sample_dataframe, config, spark_session)

def test_preprocess(data_processor):
    data_processor.preprocess()
    df = data_processor.df
    
    assert "GarageYrBlt" not in df.columns
    assert "GarageAge" in df.columns
    assert df["GarageAge"].notnull().all()
    assert df["MasVnrType"].dtype.name == "category"
    assert df["MasVnrArea"].dtype == "float64"
    assert df["LotFrontage"].notnull().all()

def test_split_data(data_processor):
    train_set, test_set = data_processor.split_data(test_size=0.5, random_state=42)
    assert len(train_set) + len(test_set) == len(data_processor.df)
    assert len(test_set) == len(train_set)

def test_save_to_catalog(mocker, data_processor):
    mock_spark_df = mocker.patch("pyspark.sql.SparkSession.createDataFrame")
    mock_spark_df.return_value.withColumn.return_value.write.mode.return_value.saveAsTable.return_value = None
    
    train_set, test_set = data_processor.split_data()
    data_processor.save_to_catalog(train_set, test_set)
    
    mock_spark_df.assert_called()

def test_enable_change_data_feed(mocker, data_processor):
    mock_sql = mocker.patch.object(data_processor.spark, "sql")
    
    data_processor.enable_change_data_feed()
    
    assert mock_sql.call_count == 2
