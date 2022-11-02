import pandas as pd
import streamlit as st
import json
from sklearn.model_selection import train_test_split

from opt_model import __XGBOOST_regression_train__

def __write_json__(new_data, variable_str, filename):
    with open(filename, 'r+') as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        # Delete previous instance
        if file_data[variable_str]:
            file_data[variable_str].pop()
        # Join new_data with file_data inside emp_details
        file_data[variable_str].append(new_data)
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        json.dump(file_data, file, indent = 4)

st.set_page_config(
     page_title="Argentina Optimization Toolkit",
     page_icon="🔍",
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

        model_path = str(st.session_state['root_dir']) + '\\model.sav'

        frames = [st.session_state['df'], retrain_df]
        df = pd.concat(frames)

        if 't_t_ratio_retrain' not in st.session_state:
            t_t_ratio_retrain = st.number_input('Select a number between 0 and 1', key='tt_ratio_r', min_value=0.0, max_value=1.0, format='%f')
            st.session_state['t_t_ratio_retrain'] = t_t_ratio_retrain
            st.session_state['train_check_retrain'] = 'False'
        else:
            t_t_ratio_retrain = st.number_input('Select a number between 0 and 1', key='tt_ratio_r', min_value=0.0, max_value=1.0, format='%f')
            st.session_state['t_t_ratio_retrain'] = t_t_ratio_retrain
            st.session_state['train_check_retrain'] = 'False'

            if st.session_state['t_t_ratio_retrain'] != 0:
                if st.button('Retrain model'):
                    st.session_state['train_check_retrain'] = 'True'

                if st.session_state['train_check_retrain'] == 'True':

                    x_train, x_test, y_train, y_test = train_test_split(df[st.session_state['feature_names']], 
                                                                            df[st.session_state['output_var']], 
                                                                            test_size=float(st.session_state['t_t_ratio_retrain']), 
                                                                            random_state=42)

                    table_str = f"""
            |**R2** | **MAE** | **MSE** |
            | :----: | :----: | :----: |"""

                    train_r2, train_mae, train_mse, test_r2, test_mae, test_mse, xgb_model = __XGBOOST_regression_train__(x_train, x_test, y_train, y_test, model_path=model_path)
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

                    st.session_state['model'] = xgb_model
                    st.session_state['model_error'][0] = train_r2
                    st.session_state['model_error'][1] = train_mae
                    st.session_state['model_error'][2] = train_mse
                    st.session_state['model_error'][3] = test_r2
                    st.session_state['model_error'][4] = test_mae
                    st.session_state['model_error'][5] = test_mse
                    params_json = str(st.session_state['root_dir']) + '\\params.json'
                    __write_json__(st.session_state['model_error'], 'model_error', params_json)
              
