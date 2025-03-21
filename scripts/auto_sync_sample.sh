#!/bin/bash

# Exit on error
set -e

echo "Building the package..."
uv build

echo "Updating lock file..."
uv lock

echo "Syncing with Databricks workspace..."
databricks sync . /Users/<your-email-address>/.bundle/dev/marvelous-databricks-course-<schema-name>

echo "Copying package to Databricks filesystem..."
databricks fs cp dist/house_price-0.0.1-py3-none-any.whl dbfs:/Volumes/<catalog-name>/<schema-name>/<package-name> --overwrite

# chmod +x scripts/auto_sync.sh 
# exe  running ./scripts/auto_sync_sample.sh