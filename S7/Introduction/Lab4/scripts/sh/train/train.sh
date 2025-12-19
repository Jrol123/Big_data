#!/bin/bash

echo 'Waiting for services to start...'
sleep 10
echo 'Starting training pipeline...'
python scripts/py/data/data_collection.py
python scripts/py/data/data_validation.py
python scripts/py/train.py
# Inference
echo ''
echo 'Starting inference pipeline...'
python scripts/py/inference.py
