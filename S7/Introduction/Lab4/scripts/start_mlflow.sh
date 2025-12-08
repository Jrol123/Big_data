#!/bin/bash
# start_mlflow.sh

echo "Waiting for PostgreSQL to be ready..."
sleep 5

echo "Starting MLflow server..."
exec mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri postgresql://mlflow:mlflow@postgres:5432/mlflowdb --default-artifact-root s3://mlflow-artifacts/ --disable-security-middleware