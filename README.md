# Optimization toolkit

## Introduction

This project provides the optimal parameters for a target output for regression problems. The ML model used is [XGBRegressor](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor), a part of [XGBoost](https://xgboost.readthedocs.io/en/stable/) which is a [popular](https://github.com/dmlc/xgboost/tree/master/demo#machine-learning-challenge-winning-solutions) optimized distributed gradient boosting library. This toolkit also is capable of providing output predictions and retrainig the model. The algorithm for finding the optimal parameters is simulated annealing, using [dual annealing](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html) from [scipy](https://docs.scipy.org/doc/scipy/index.html).

The user interface was built using [streamlit](https://streamlit.io/), but the parts that contain ML methods are separate from the UI, so it is possible to use the functions with other methods. MUDAR ESTA FRASE QUE ESTÁ MUTIO FEIA

## Main features

- **Output prediction**: after training the model, insert a .csv file containing the values of each feature. BALKBDAKSHBDSAHDBASD
- **Feature optimization**: define each feature type (continuous or discrete) and its range (which can also be a fixed value) and train the model. Insert a .csv file containing the current values for each parameter, choose a target output and the program will find the best value for each parameter in order to get the closest output value to the target.
- **Model explanation using [SHAP](https://shap.readthedocs.io/en/latest/index.html)**: the XGBRegressor model isn't easy to decode and is known as a *black box model*. SHAP helps explain the importance of each feature to the model's output, using an approach relying on Game Theory.  

## Use cases

This toolkit is built for [regression problems](https://en.wikipedia.org/wiki/Regression_analysis) using categorical data, with either continuous or discrete values. It is still possible to use nominal data, but each category must be attributed to a number and before training the model, this data must be described as *discrete* in the *Parameter Definition* tab of the app. The toolkit only functions with datasets that have had the possible specified modifications made and by being in a ***.csv*** file.

This project was built using this [dataset COLOCAR AQUI O BLEACH CENAS]() which 

## Step-by-step guide

The python packages required to run the toolkit are: [sklearn](https://scikit-learn.org/stable/), [SciPy](https://scipy.org/), [pandas](https://pandas.pydata.org), [NumPy](https://numpy.org/), [Streamlit](https://streamlit.io/), [matplotlib](https://matplotlib.org/), [SHAP](https://shap.readthedocs.io/en/latest/index.html), [XGBoost](https://xgboost.readthedocs.io/en/stable/) and [Hyperopt](http://hyperopt.github.io/hyperopt/).

To install these packages run:
- pip install scikit-learn
- pip install scipy
- pip install pandas
- pip install numpy
- pip install matplotlib
- pip install shap
- pip install xgboost
- pip install hyperopt

To run the toolkit:
- streamlit run 🔍_Initial_Page.py

A new browser window should appear as shown. The menu bar contains sections to process your dataset. 
![imagem](https://user-images.githubusercontent.com/78873306/199540588-ff62123e-918c-4d21-b2fa-0feefd723777.png)
The *Initial Page* 

## Internal structure

The main reason for so many different files is the way streamlit [works](https://docs.streamlit.io/library/get-started/multipage-apps). All functions related to ML are separate from the functions which relate to streamlit in order to be easier to migrate the functions to a different GUI.

#### Non-code files

The toolkit stores the trained model and several other features, such as which variables are discrete, the minimal increment of each and the original dataframe. To store these, 4 JSON files are created after defining the model's parameters for the first time. A time-saving measure was taken while writing the code and the program simply copies a template and fills it in with the model's parameters, that's why a *template.json* file is in the directory.

#### XGBoost Regression

The function used is [XGBRegressor](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRegressor). Since this toolkit is meant to be used with generic regression problems, the model's hyperparameters are adjusted using [hyperopt](http://hyperopt.github.io/hyperopt/).

#### Parameter optimization

Simulated annealing through [SciPy](https://docs.scipy.org/doc/scipy/index.html)'s [dual_annealing](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html) is used to get the parameters values for which the ML model predicts the closest value to the target output. This optimization process takes into account boundaries for each variable, with those being either imposed by the user (as an interval or as a fixed value) or assumed to be the minimum and maximum values of the variable available in the dataset.

The [dual_annealing](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html) function has an argument called callback for which is possible to create a function that display each iteration of the algorithm to see how the variables converge. As this is not critical to the goal of this toolkit, it has not been implemented.

#### Discrete variable optimization

At the time of writing, [dual_annealing](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html) is not capable of restricting the variables' values into discrete ones (and probably never will), so a different strategy must be followed. The toolkit implementation to handle discrete values is very simple, it just takes the results from the dual_annealing, rounds them to the nearest selected discrete step and predicts the new output value. This approach will not yield the best possible results, but through testing using different datasets, the findings showed that the results achieved using this method provided marginal diferences to the predicted output with the continuous variables' values obtained through the dual_annealing. This might be due to the always present inaccuracy of the ML model, which might not be very susceptible to slight changes in the parameters. It is also important to note that the output achieved by [mixed-integer programming](https://en.wikipedia.org/wiki/Integer_programming) (another term for this type of optimization) can never be closer to the goal than its continuous counterpart. The opposite of this fact might happen using this method due to the error associated with the ML model, although this has yet to be observed.

A more appropriate strategy for discrete variable optimization would be to use [mixed-integer programming](https://en.wikipedia.org/wiki/Integer_programming) algorithms, through [PuLP](https://coin-or.github.io/pulp/main/optimisation_concepts.html#integer-programing) or some other optimization framework. Although these algorithms might be the better approach from a purely analytical standpoint, it has more drawbacks in the context of this toolkit than it would normally have. The standard disadvantage is the computational power needed for this type of programming and the contextual one is the much discussed inaccuracy of the ML model, which is the objective function for these algorithms, which depending on how big this error is it might not be worth it sacrificing efficiency for possibly inaccurate results.

#### Model retraining

To achieve faster and possibly more accurate retraining, [incremental learning](https://en.wikipedia.org/wiki/Incremental_learning) is used. This method has already been implemented in [XGBRegressor](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor.fit) and is achieved by saving the ML model once it has been trained for a first time and while fitting the new data, it is only needed to pass the argument `xgb_model=`, giving it the saved model (which in this toolkit is saved as a JSON file). 

#### Model parameter visualization

XGBoost is considered to be a black box model, in that the user has no overview of how each of the model's parameters affects the final result, given the model's high complexity. As mentioned previously, [SHAP](https://shap.readthedocs.io/en/latest/index.html) is used to help understand the importance of each feature to the model's output. At the time of writing, only a [beeswarm plot](https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/beeswarm.html) is implemented which is designed to display an information-dense summary of how the top features in a dataset impact the model’s output and is quite intuitive. However, caution is needed as SHAP helps to show the correlations picked up by predictive ML models, which is not the same as causation, as detailed in [this](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/Be%20careful%20when%20interpreting%20predictive%20models%20in%20search%20of%20causal%C2%A0insights.html) overview in the SHAP documentation

#### Streamlit struggles

[Streamlit](https://streamlit.io/) is a Pyhton package intended to simplify building data apps. It's a great tool to make simple apps quickly, however, due to its simplicity, it has a few quirks. The most noteacible one in this toolkit is the fact it reruns the whole Python script if an action is performed, for example, a button press. So, how is it possible to save a variable after rerunning the script over and over? The answer is [*st.session_state*](https://docs.streamlit.io/library/api-reference/session-state), which is the way to store variables/states in between reruns for each user session. The state is preserved until the user reloads the app. This makes it so that sometimes it is necessary to press a button 2 times for the next inteded step to show, but please be aware that it does not affect the data analytics part of this toolkit as the most that could happen is to rerun an algorithm and having to wait for a few seconds while it finishes properly. 

## Possible improvements

- **[Incremental learning](https://en.wikipedia.org/wiki/Incremental_learning):** Retraining the model without reusing the old data, which improves time efficiency
- **More plots for easier interpretation**: [shapley values](https://en.wikipedia.org/wiki/Shapley_value) are not as [straightforward](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html#The-additive-nature-of-Shapley-values) as one might think. SHAP helps to show the correlations picked up by predictive ML models, which is not the same as causation, as detailed in [this](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/Be%20careful%20when%20interpreting%20predictive%20models%20in%20search%20of%20causal%C2%A0insights.html) overview in the SHAP documentation 
- **[XGBoost feature importance](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.Booster.get_score)**: complement SHAP with interpretation tools built-in XGBoost
- **Convergence plots**: while the [dual_annealing](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html) function is running, it is possible to have another function running to see how the convergence process occurs. At this moment, this function prints the variable values to the terminal but this could be improved by plotting the variables to a graph. It is important to note that this would not change the final values
- **GPU support**
