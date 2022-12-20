#!/bin/bash

echo "running Adaboost"
python3 credAdaBoost.py

echo "running Neural net initial implenetation"
python3 simpleNet.py

echo "running Neural net modified version"
python3 NeuralNet.py

echo "running Logistic Regression"
python3 logistic.py