import pandas as pd
import streamlit as st
import pickle
import os

from select_model import __LINEAR_regression_train__, __SGD_regression_train__, __DECISION_TREE_regression_train__, __XGBOOST_regression_train__
from train_model import __train_model__, __plot_decision__, __increment_training__

st.set_page_config(
     page_title="Argentina Optimization Toolkit",
     page_icon="🤡",
     initial_sidebar_state="auto"
 )

st.session_state['page_change'] = True

st.title('Model retraining')

if 'model' not in st.session_state:
    st.write('A model must be selected before retraining it')
else:
    model = st.session_state['model']
    retrain_file = st.file_uploader('Choose a .csv containing the new values to retrain the model')

    st.markdown('----')

    if retrain_file is not None:
        if 'retrain_df' not in st.session_state:
            retrain_df =  pd.read_csv(retrain_file)
            st.session_state['retrain_df'] = retrain_df
        else:
            retrain_df = st.session_state['retrain_df']

        x_train = pd.concat([st.session_state['x_train'], retrain_df[st.session_state['feature_names']]])
        y_train = pd.concat([st.session_state['y_train'], retrain_df[st.session_state['output_var']]])
        x_test = st.session_state['x_test']
        y_test = st.session_state['y_test']

        table_str = f"""
|**R2** | **MAE** | **MSE** |
| :----: | :----: | :----: |"""
            
        train_r2, train_mae, train_mse, test_r2, test_mae, test_mse, xgb_model = __XGBOOST_regression_train__(x_train, x_test, y_train, y_test)
        st.markdown('#### Training data')
        train_str = f"""
|{str('{:10.4f}'.format(train_r2))} | {str('{:10.4f}'.format(train_mae))} | {str('{:10.4f}'.format(train_mse))} |"""
        full_table_str = table_str + train_str
            
        st.markdown(full_table_str)

        st.markdown('#### Testing data')
        test_str = f"""
|{str('{:10.4f}'.format(test_r2))} | {str('{:10.4f}'.format(test_mae))} | {str('{:10.4f}'.format(test_mse))} |"""
        full_table_str = table_str + test_str
        st.markdown(full_table_str)
        st.text("")
        
        if st.button('Save model'):
            model_file = str(st.session_state['root_dir']) + '\\model.sav'
            os.remove(model_file)
            pickle.dump(xgb_model, open(model_file, 'wb'))

            st.session_state['model_error'][0] = train_r2
            st.session_state['model_error'][1] = train_mae
            st.session_state['model_error'][2] = train_mse
            st.session_state['model_error'][3] = test_r2
            st.session_state['model_error'][4] = test_mae
            st.session_state['model_error'][5] = test_mse