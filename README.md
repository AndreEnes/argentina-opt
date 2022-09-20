# Optimization toolkit

## Introduction

This project provides the optimal parameters for a target output for regression problems. The ML model used is [XGBRegressor](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor), a part of [XGBoost](https://xgboost.readthedocs.io/en/stable/) which is a [popular](https://github.com/dmlc/xgboost/tree/master/demo#machine-learning-challenge-winning-solutions) optimized distributed gradient boosting library. This toolkit also is capable of providing output predictions and retrainig the model. The algorithm for finding the optimal parameters is simulated annealing, using [dual annealing](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html) from [scipy](https://docs.scipy.org/doc/scipy/index.html).

The user interface was built using [streamlit](https://streamlit.io/), but the parts that contain ML methods are separate from the UI, so it is possible to use the functions with other methods. MUDAR ESTA FRASE QUE ESTÁ MUTIO FEIA

## Main features

## Example usage

## Step-by-step guide

## Internal structure

## Possible improvements

[Incremental learning](https://en.wikipedia.org/wiki/Incremental_learning) -- Retraining the model without reusing the old data, which improves time efficiency
