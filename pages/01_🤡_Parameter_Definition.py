import streamlit as st
import warnings
from tqdm import tqdm 
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import pickle
import os
import json
from opt_model import __XGBOOST_regression_train__, __train_model__

warnings.filterwarnings("ignore")
tqdm.pandas()


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
     page_icon="🤡",
     initial_sidebar_state="auto"
 )

st.title("Optimization toolkit")


uploaded_file = st.file_uploader("Choose a .csv file")

if uploaded_file is not None:
    if 'initial_file' not in st.session_state:
        df = pd.read_csv(uploaded_file)
        if 'df' not in st.session_state:
            st.session_state['df'] = df
        st.session_state['initial_file'] = uploaded_file
    else:
        if st.button('Change file'):
            df = pd.read_csv(uploaded_file)
            st.session_state['df'] = df


if 'df' in st.session_state:
    df = st.session_state['df']

    if (('output_var' and 'feature_names') not in st.session_state):
        output_var = 'invalid'
        selected_output_var = st.selectbox('Select the output variable', df.columns)

        if st.button('Select', key='select0'):
            output_var = selected_output_var        
            feature_names = df.columns.to_list()
            feature_names.remove(output_var)
            df_no_output = df[feature_names]
            st.session_state['output_var'] = output_var
            st.session_state['feature_names'] = feature_names
            st.session_state['df_no_output'] = df_no_output

    else:
        output_var = st.session_state['output_var']
        feature_names = st.session_state['feature_names']
        df_no_output = st.session_state['df_no_output']
        selected_output_var = st.selectbox('Select the output variable', df.columns)

        if st.button('Select', key='select0'):
            output_var = selected_output_var
            feature_names = df.columns.to_list()
            feature_names.remove(output_var)
            df_no_output = df[feature_names]
            st.session_state['output_var'] = output_var
            st.session_state['feature_names'] = feature_names
            st.session_state['df_no_output'] = df_no_output


    #each line of df_vars contains the name of a variable, its numerical limits, whether its discrete or not and the minimal increment
    if 'df_vars' not in st.session_state:
        df_vars = []
        for i in df:
            df_vars.append([df[i].name, None, None, None])
        st.session_state['df_vars'] = df_vars
    else:
        df_vars = st.session_state['df_vars']

    if output_var != 'invalid':
        st.markdown('----')
        st.subheader('Discrete variables')

        if 'discrete_vars' not in st.session_state:    
            discrete_vars = st.multiselect('Select discrete variables', df.columns)
            st.session_state['discrete_vars'] = discrete_vars
            st.session_state['df_vars'] = df_vars
        else:
            df_vars = st.session_state['df_vars']
            discrete_vars = st.multiselect('Select discrete variables', df.columns)

            if discrete_vars:
                for i in range(len(discrete_vars)):
                    for j in range(len(df_vars)):
                        if discrete_vars[i] == df_vars[j][0]:
                            df_vars[j][2] = 'DISCRETE'

                if len(discrete_vars) < len(st.session_state['discrete_vars']):
                    s = set(discrete_vars)
                    unselected_var = [x for x in st.session_state['discrete_vars'] if x not in s]   #https://stackoverflow.com/questions/3462143/get-difference-between-two-lists
                    for i in range(len(discrete_vars)):
                        for j in range(len(df_vars)):
                            if unselected_var[0] == df_vars[j][0]:
                                df_vars[j][2] = None
            else:
                for i in range(len(df_vars)):
                        df_vars[i][2] = None

            st.session_state['discrete_vars'] = discrete_vars
            df_vars = st.session_state['df_vars']
            
    if 'discrete_vars' in st.session_state:
        df_vars = st.session_state['df_vars']
        st.markdown('----')
        st.markdown('## Minimal increment')
        st.markdown('If left blank, the variable is assumed to be an integer')
        increment = []

        for i,v in enumerate(df_vars):
            if v[2] == 'DISCRETE':
                var_str = v[0] + ' minimal increment'
                increment_input = st.number_input(var_str, key='inc' + str(i), step=0.00001, format='%f')
                increment.append(increment_input)
        
        st.session_state['increment'] = increment
        
    if 'increment' in st.session_state:
        df_vars = st.session_state['df_vars']
        aux = 0

        for i,v in enumerate(df_vars):
            if v[2] == 'DISCRETE':
                v[3] = increment[aux]
                aux += 1

        st.session_state['df_vars'] = df_vars

    if output_var != 'invalid':
        st.markdown('----')
        st.markdown('## Variable ranges')
        st.markdown('If left blank, the range of the variable is assumed to be the minimum and maximum in the dataset provided')
        ranges_var = st.selectbox('Select variable', df_no_output.columns)

        if 'ranges_var' not in st.session_state:
            st.session_state['ranges_var'] = ranges_var
        else:
            if 'choose_button' not in st.session_state:
                choose_button = st.button('Choose range')
                st.session_state['choose_button'] = choose_button
                st.session_state['range_select'] = 'True'
            else:
                choose_button = st.button('Choose range')
                st.session_state['choose_button'] = choose_button
                if st.session_state['choose_button'] and st.session_state['range_select'] == 'False':
                    st.session_state['range_select'] = 'True'                
                if 'range_select' not in st.session_state:
                    range_select = st.radio('Choose range type:', ('Fixed number', 'Range'))
                    st.session_state['range_select'] = 'True'
                elif st.session_state['range_select'] == 'True':
                    range_select = st.radio('Choose range type:', ('Fixed number', 'Range'))
                    if range_select == 'Fixed number':
                        range_fixed = st.number_input('Select a fixed number', format='%f')
                        if st.button('Submit'):
                            for i in range(len(df_vars)):
                                if ranges_var == df_vars[i][0]:
                                    df_vars[i][1] = range_fixed
                            st.session_state['range_select'] = 'False'
                    elif range_select == 'Range':
                        with st.form(key='form'):
                            c1, c2 = st.columns(2)
                            with c1:
                                limit_min = st.number_input('Minimum', key='min', format='%f')
                            with c2:
                                limit_max = st.number_input('Maximum', key='max', format='%f')
                            submit_button = st.form_submit_button(label='Submit')

                            if submit_button:
                                for i in range(len(df_vars)):
                                    if ranges_var == df_vars[i][0]:
                                        df_vars[i][1] = [limit_min, limit_max]

                                st.session_state['range_select'] = 'False'
                
        st.session_state['df_vars'] = df_vars


    if output_var != 'invalid':
        st.markdown('----')
        st.markdown('## Ratio between training and testing data')

        if 't_t_ratio' not in st.session_state:
            t_t_ratio = st.number_input('Select a number between 0 and 1', key='tt_ratio', min_value=0.0, max_value=1.0, format='%f')
            st.session_state['t_t_ratio'] = t_t_ratio
            st.session_state['train_check'] = 'False'
        else:
            t_t_ratio = st.number_input('Select a number between 0 and 1', key='tt_ratio', min_value=0.0, max_value=1.0, format='%f')
            st.session_state['t_t_ratio'] = t_t_ratio
            st.session_state['train_check'] = 'False'
            
            df = st.session_state['df']
            
            if st.session_state['t_t_ratio'] != 0:
                if st.button('Train model'):
                    st.session_state['train_check'] = 'True'

                if 'model' in st.session_state and st.session_state['train_check'] == 'True':
                    st.markdown('----')
                    st.markdown('# Model')

                    x_train, x_test, y_train, y_test = train_test_split(df[feature_names], 
                                                                df[output_var], 
                                                                test_size=float(t_t_ratio), 
                                                                random_state=42)
                    
                    st.session_state['x_train'] = x_train
                    st.session_state['y_train'] = y_train
                    st.session_state['x_test'] = x_test
                    st.session_state['y_test'] = y_test
                    
                    table_str = f"""
|**R2** | **MAE** | **MSE** |
| :----: | :----: | :----: |"""

                    st.markdown('## XGBoost Regression')   #FICA FEIO ASSIM, MAS CENAS
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

                    if 'model_error' not in st.session_state:
                            st.session_state['model_error'] = []
                            
                    st.session_state['model_error'].append(train_r2)
                    st.session_state['model_error'].append(train_mae)
                    st.session_state['model_error'].append(train_mse)
                    st.session_state['model_error'].append(test_r2)
                    st.session_state['model_error'].append(test_mae)
                    st.session_state['model_error'].append(test_mse)
                    
                    model = xgb_model
                    st.session_state['model_name'] = 'XGBoost Regression'

                    st.session_state['model'] = model
                    #https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
                    model_file = str(st.session_state['root_dir']) + '\\model.sav'
                    pickle.dump(model, open(model_file, 'wb'))
                    curr_dir = os.getcwd()
                    curr_dir_file_1 = curr_dir + '\\' + model_file
                    
                    json_file = str(st.session_state['root_dir']) + '\\params.json'
                    __write_json__(df_vars, 'df_vars', json_file)
                    __write_json__(feature_names, 'feature_names', json_file)
                    __write_json__(output_var, 'output_var', json_file)
                    __write_json__(st.session_state['model_name'], 'model_name', json_file)
                    __write_json__(st.session_state['t_t_ratio'], 't_t_ratio', json_file)
                    
                    df_pickle_str = str(st.session_state['root_dir']) + '\\df.pkl' 
                    df_no_output_pickle_str = str(st.session_state['root_dir']) + '\\df_no_output.pkl'
                    df.to_pickle(df_pickle_str)
                    df_no_output.to_pickle(df_no_output_pickle_str)
                elif 'model' not in st.session_state and st.session_state['train_check'] == 'True':
                        x_train, x_test, y_train, y_test = train_test_split(df[feature_names], 
                                                                    df[output_var], 
                                                                    test_size=float(t_t_ratio), 
                                                                    random_state=42)

                        st.session_state['x_train'] = x_train
                        st.session_state['y_train'] = y_train
                        st.session_state['x_test'] = x_test
                        st.session_state['y_test'] = y_test

                        table_str = f"""
|**R2** | **MAE** | **MSE** |
| :----: | :----: | :----: |"""

                        st.markdown('## XGBoost Regression')
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

                        if 'model_error' not in st.session_state:
                            st.session_state['model_error'] = []

                        st.session_state['model_error'].append(train_r2)
                        st.session_state['model_error'].append(train_mae)
                        st.session_state['model_error'].append(train_mse)
                        st.session_state['model_error'].append(test_r2)
                        st.session_state['model_error'].append(test_mae)
                        st.session_state['model_error'].append(test_mse)

                        st.session_state['train_check'] = 'False'
                        model = xgb_model
                        st.session_state['model_name'] = 'XGBoost Regression'
    
                        st.session_state['model'] = model
                        #https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
                        model_file = str(st.session_state['root_dir']) + '\\model.sav'
                        pickle.dump(model, open(model_file, 'wb'))
                        curr_dir = os.getcwd()
                        curr_dir_file_1 = curr_dir + '\\' + model_file
                        
                        json_file = str(st.session_state['root_dir']) + '\\params.json'
                        __write_json__(df_vars, 'df_vars', json_file)
                        __write_json__(feature_names, 'feature_names', json_file)
                        __write_json__(output_var, 'output_var', json_file)
                        __write_json__(st.session_state['model_name'], 'model_name', json_file)
                        __write_json__(st.session_state['t_t_ratio'], 't_t_ratio', json_file)
                        
                        df_pickle_str = str(st.session_state['root_dir']) + '\\df.pkl' 
                        df_no_output_pickle_str = str(st.session_state['root_dir']) + '\\df_no_output.pkl'
                        df.to_pickle(df_pickle_str)
                        df_no_output.to_pickle(df_no_output_pickle_str)

            elif st.session_state['t_t_ratio'] == 0:
                st.write('0 is not a valid ratio')

