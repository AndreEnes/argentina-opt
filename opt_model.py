import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
#Import 'scope' from hyperopt in order to obtain int values for certain hyperparameters.
from hyperopt.pyll.base import scope     

def __train_model__(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    return train_r2, train_mae, train_mse, test_r2, test_mae, test_mse

def __XGBOOST_regression_train__(x_train, x_test, y_train, y_test):

    #Define the space over which hyperopt will search for optimal hyperparameters.
    space = {'max_depth': scope.int(hp.quniform("max_depth", 1, 5, 1)),
            'gamma': hp.uniform ('gamma', 0,1),
            'reg_alpha' : hp.uniform('reg_alpha', 0,50),
            'reg_lambda' : hp.uniform('reg_lambda', 10,100),
            'colsample_bytree' : hp.uniform('colsample_bytree', 0,1),
            'min_child_weight' : hp.uniform('min_child_weight', 0, 5),
            'n_estimators': 10000,
            'learning_rate': hp.uniform('learning_rate', 0, .15),
            'random_state': 5,
            'max_bin' : scope.int(hp.quniform('max_bin', 200, 550, 1))}        

    #Define the hyperopt objective.
    def hyperparameter_tuning(space):
        model = xgb.XGBRegressor(**space)
        
        #Define evaluation datasets.
        evaluation = [(x_train, y_train), (x_test, y_test)]
        
        #Fit the model. Define evaluation sets, early_stopping_rounds, and eval_metric.
        model.fit(x_train, y_train,
                eval_set=evaluation, eval_metric="rmse",
                early_stopping_rounds=100,verbose=False)
    
        #Obtain prediction and rmse score.
        pred = model.predict(x_test)
        rmse = mean_squared_error(y_test, pred, squared=False)
        print ("SCORE:", rmse)
        
        #Specify what the loss is for each model.
        return {'loss':rmse, 'status': STATUS_OK, 'model': model}

    #Run 20 trials.
    trials = Trials()
    best = fmin(fn=hyperparameter_tuning,
                space=space,
                algo=tpe.suggest,
                max_evals=30,
                trials=trials)
    
    #Create instace of best model.
    best_model = trials.results[np.argmin([r['loss'] for r in 
    trials.results])]['model']

    xgb_train_r2, xgb_train_mae, xgb_train_mse, xgb_test_r2, xgb_test_mae, xgb_test_mse = __train_model__(best_model, x_train.values, y_train.values, x_test.values, y_test.values)

    return xgb_train_r2, xgb_train_mae, xgb_train_mse, xgb_test_r2, xgb_test_mae, xgb_test_mse, best_model