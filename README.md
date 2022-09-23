# Optimization toolkit

## Introduction

This project provides the optimal parameters for a target output for regression problems. The ML model used is [XGBRegressor](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor), a part of [XGBoost](https://xgboost.readthedocs.io/en/stable/) which is a [popular](https://github.com/dmlc/xgboost/tree/master/demo#machine-learning-challenge-winning-solutions) optimized distributed gradient boosting library. This toolkit also is capable of providing output predictions and retrainig the model. The algorithm for finding the optimal parameters is simulated annealing, using [dual annealing](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html) from [scipy](https://docs.scipy.org/doc/scipy/index.html).

The user interface was built using [streamlit](https://streamlit.io/), but the parts that contain ML methods are separate from the UI, so it is possible to use the functions with other methods. MUDAR ESTA FRASE QUE ESTÁ MUTIO FEIA

## Main features

- **Output prediction**: after training the model, insert a .csv file containing the values of each feature. BALKBDAKSHBDSAHDBASD
- **Feature optimization**: define each feature type (continuous or discrete) and its range (which can also be a fixed value) and train the model. Insert a .csv file containing the current values for each parameter, choose a target output and the program will find the best value for each parameter in order to get the closest output value to the target.
- **Model explanation using [SHAP](https://shap.readthedocs.io/en/latest/index.html)**: the XGBRegressor model isn't easy to decode and is known as a *black box model* (REBER ESTA FRASE PARA VER SE NÃO ESTOU A INVENTAR). SHAP helps explain the importance of each feature to the model's output, using an approach relying on Game Theory.  

## Example usage

## Step-by-step guide

## Internal structure

The main reason for so many different files is the way streamlit [works](https://docs.streamlit.io/library/get-started/multipage-apps). All functions related to ML are separate from the functions which relate to streamlit in order to be easier to migrate the functions to a different GUI.

#### Non-code files

The toolkit stores the trained model and several other features, such as which variables are discrete, the minimal increment of each and the original dataframe. To store these, 4 JSON files are created after defining the model's parameters for the first time. A time-saving measure was taken while writing the code and the program simply copies a template and fills it in with the model's parameters, that's why a *template.json* file is in the directory.

#### XGBoost Regression

#### Parameter optimization

#### Discrete variable optimization

#### Model retraining

#### Model parameter visualization

#### Streamlit struggles

## Possible improvements

- **[Incremental learning](https://en.wikipedia.org/wiki/Incremental_learning):** Retraining the model without reusing the old data, which improves time efficiency
- **More plots for easier interpretation**: [shapley values](https://en.wikipedia.org/wiki/Shapley_value) are not as [straightforward](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html#The-additive-nature-of-Shapley-values) as one might think. SHAP helps to show the correlations picked up by predictive ML models, which is not the same as causation, as detailed in [this](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/Be%20careful%20when%20interpreting%20predictive%20models%20in%20search%20of%20causal%C2%A0insights.html) overview in the SHAP documentation 
- **[XGBoost feature importance](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.Booster.get_score)**: complement SHAP with interpretation tools built-in XGBoost
- **Convergence plots**: while the [dual_annealing](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html) function is running, it is possible to have another function running to see how the convergence process occurs. At this moment, this function prints the variable values to the terminal but this could be improved by plotting the variables to a graph. It is important to note that this would not change the final values
- **GPU support**
