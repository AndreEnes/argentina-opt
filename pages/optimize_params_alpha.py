from statistics import mode
from tkinter.tix import Tree
from sklearn.metrics import mean_squared_error
from scipy.optimize import dual_annealing, minimize

import numpy as np
import streamlit as st
import shap
import pandas as pd
import json

shap.initjs()

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

def __roundPartial__(value, resolution):
    return round (value / resolution) * resolution

# Python code to sort the tuples using second element 
# of sublist Inplace way to sort using sort()
def __sort_2nd_element__(sub_li):
  
    # reverse = None (Sorts in Ascending order)
    # key is set to sort using second element of 
    # sublist lambda has been used
    sub_li.sort(key = lambda x: -x[1]) #the minus turns the list into descending order
    return sub_li

def fitness_function(x, target_output, features_space):
    model = st.session_state['model']
    # concats the fixed variables with the non fixed variables
    x_concat = build_feature_array(x, features_space)
    #if st.session_state['model_name'] == 'XGBoost Regression':
    #    st.session_state['hhd'] = [x_concat]
    #st.write('XGB x_concat --> ', st.session_state['hhd'])
    #st.write('bla', [x_concat])
    # get the value of output for the current solution
    curr_value = model.predict([x_concat])
    #st.write('curr_value', curr_value)
    # computes the error between the current output and the target
    return mean_squared_error(curr_value, [target_output])

def build_feature_array(x, features_space):
    x_concat = np.zeros(len(features_space))
    x_list = list(x)
    for i, v in enumerate(features_space):
        # appends the fixed feature values
        if type(v[1]) != tuple:
            x_concat[i] = v[1]
        # appends the non fixed feature values
        else:
            x_concat[i] = x_list.pop(0)
    # returns the results
    return x_concat

def dual_annealing_callback(x, f, context):
    print('non-fixed params: {0}, whiteness: {1}'.format(x.round(2), f.round(3)))
    
def minimize_callback(xk):
    print(xk)


def __optimize_params__(x0, model, explainer, target_output, features_space, df, cb=dual_annealing_callback):
    discrete_flag = False

    # creates boundaries for the not defined variables
    for i, v in enumerate(features_space):
        if v[1] is None:
            features_space[i][1] = (df[v[0]].min(), df[v[0]].max())
        if v[2] == 'DISCRETE':
            discrete_flag = True
    # configures the x0 according to the boundaries
    nff_idx, bounds = zip(*[(i, v[1]) for i, v in enumerate(features_space) if type(v[1]) == tuple])
    x0_filtered = [v for i, v in enumerate(x0) if i in set(nff_idx)]
    # optimization
    res = dual_annealing(fitness_function, 
                         bounds, 
                         x0=x0_filtered, 
                         callback=cb, 
                         args=[target_output, features_space], 
                         maxfun=1e3,
                         seed=16)
    
    # res = minimize(fitness_function, x0_filtered, 
    #                method='Powell', 
    #                bounds=bounds, 
    #                callback=minimize_callback, 
    #                args=(target_whiteness, features_space),
    #                options={'maxiter': 5000, 'disp': True},
    #                tol=1e-6)
    # gest the best params and mse 
    best_params = build_feature_array(res.x, features_space)
    mse = res.fun

    if discrete_flag:
        best_params, mse = __discrete_params_optimization__(features_space, explainer, x0, target_output, model)

    # returns the results
    return best_params, mse

def __discrete_params_optimization__(features_space, explainer, x0, target_output, model, cb=dual_annealing_callback): #procurar Branch&Bound
    discrete_vars = []
    x0_columns = []
    
    for i,v in enumerate(features_space):
        x0_columns.append(v[0]) 
    
    x0_df = pd.DataFrame(x0)
    x0_df = x0_df.transpose()
    x0_df.columns = x0_columns

    #get shapley values for this particular set of values
    shap_values = explainer.shap_values(x0_df)    #PROBLEMA
    #attribute shapley values to the discrete variables
    for i, v in enumerate(features_space):
        if v[2] == 'DISCRETE':
            if v[3] != 0:
                discrete_vars.append([v[0], shap_values[0][i], v[3]])
            else:
                 discrete_vars.append([v[0], shap_values[0][i], 1])
    #the higher the shapley value, the greater influence in the output
    __sort_2nd_element__(discrete_vars) #descending order by shap_values    #BHAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    #st.write('shap_values', shap_values, 'discrete_vars', discrete_vars)
    aux_array = []
    #find the discrete variable with the highest shap value in the feature space
    for i,v in enumerate(discrete_vars):                
        #aux_array.append(features_space.index(v[0]))
        x = [x for x in features_space if v[0] in x][0]
        aux_array.append([features_space.index(x), v[2]])
    
    # configures the x0 according to the boundaries
    nff_idx, bounds = zip(*[(i, v[1]) for i, v in enumerate(features_space) if type(v[1]) == tuple])
    x0_filtered = [v for i, v in enumerate(x0) if i in set(nff_idx)]
    # optimization
    best_res = dual_annealing(fitness_function,
                         bounds, 
                         x0=x0_filtered, 
                         callback=cb, 
                         args=[target_output, features_space], 
                         maxfun=1e3,
                         seed=16)
    
    params = build_feature_array(best_res.x, features_space)
    #st.write(params,'pred antes de por discrete', model.predict([params]))
    #params[aux_array[0]] = np.round(params[aux_array[0]])  #round the feature with the highest shap value
    for i,v in enumerate(aux_array):                            #BRANCH&BOUND
        if v[1] != 1:
            params[v[0]] = np.round(params[v[0]], len(str(v[1])) - 2) #não funciona assim, seu malandreco
            params[v[0]] = __roundPartial__(params[v[0]], v[1])
        else:
            params[v[0]] = np.round(params[v[0]])
        

    #best_res_pred = model.predict([params])
    #st.write('params antes de build feature cenas', params, 'pred depois de discrete', best_res_pred)
#    #TÁ QUASEEEEEEE, MAS AINDA ESTÁ UM BOCADO ESTÚPIUDOOOOOOOOOOOOOOOO
#    if len(discrete_vars) > 1:
#        for i,v in enumerate(discrete_vars):
#            try:
#                round(params[aux_array[i]])
#            except:
#                break
#            curr_res = dual_annealing(fitness_function, 
#                         bounds, 
#                         x0=x0_filtered, 
#                         callback=cb, 
#                         args=[target_output, features_space], 
#                         maxfun=1e3,
#                         seed=16)
#
#            params = build_feature_array(curr_res.x, features_space)
#            params[aux_array[i]] = np.round(params[aux_array[i]])
#            curr_res_pred = model.predict([params])
#            st.write('discrete cenas params', params)
#            best_res_diff = abs(target_output - best_res_pred)
#            curr_res_diff = abs(target_output - curr_res_pred)
#
#            if curr_res_diff < best_res_diff:   #the closest value to the output is the best option
#                best_res = curr_res

    
    #params = build_feature_array(best_res.x, features_space)
    #st.write('return da __discrete_cenas: ', params)
    mse = best_res.x

    return params, mse

    


def __streamlit_get_shap_explainer__(n_samples, model, model_name):
    x_train = pd.concat([st.session_state['x_train'], st.session_state['x_test']])
    #https://shap.readthedocs.io/en/latest/generated/shap.explainers.Sampling.html
    if model_name == ('XGBoost Regression' or 'Decision Tree'):
        explainer = shap.TreeExplainer(model, x_train[:n_samples])
    else:
        #KernelExplainer returns an approximation of the shap values, but works with any kind of model
        explainer = shap.KernelExplainer(model.predict, x_train[:n_samples], link="identity")

    return explainer


st.set_page_config(
     page_title="Argentina Optimization Toolkit",
     page_icon="🤡",
     initial_sidebar_state="auto"
 )

st.session_state['page_change'] = True

st.header('Optimize parameters')

if 'df_vars' not in st.session_state:
    st.write('No variables uploaded at the moment')    
elif 'model' not in st.session_state:
    st.write('Choose a model before accessing this page')
else:
    optimize_file = st.file_uploader('Choose a .csv file containing the feature values to be optimized')
 
    if optimize_file is not None:
        if 'opt_values_df' not in st.session_state:
            opt_values_df = pd.read_csv(optimize_file)
            st.session_state['opt_values_df'] = opt_values_df
        else:
            if st.button('Change file'):
                opt_values_df = pd.read_csv(optimize_file)
                st.session_state['opt_values_df'] = opt_values_df
        st.session_state['optimize_file'] = optimize_file
    
    if 'target_output' not in st.session_state:
        target_output = st.number_input('Select the target output')
        st.session_state['target_output'] = target_output
    else:
        target_output = st.number_input('Select the target output')
        st.session_state['target_output'] = target_output
    
        if 'shap_explainer' not in st.session_state:
            shap_explainer = __streamlit_get_shap_explainer__(154, st.session_state['model'], st.session_state['model_name'])
            st.session_state['shap_explainer'] = shap_explainer
        else:
            df_vars = st.session_state['df_vars']
            output_var = st.session_state['output_var']
            model = st.session_state['model']
            output_index = 0

            if 'df_vars_cleaned' not in st.session_state:
                for i,v in enumerate(df_vars):
                    if v[0] == output_var:
                        output_index = i        
                df_vars.remove(df_vars[output_index])
                df_vars_cleaned = df_vars.copy()
                st.session_state['df_vars_cleaned'] = df_vars_cleaned
                curr_pred = model.predict(st.session_state['opt_values_df'].iloc[0:1])
                #curr_pred_str = str('{:10.4f}'.format(curr_pred))
                #curr_pred_str = '*Current prediction*: ' + str(curr_pred)
                #st.markdown(curr_pred_str)
                curr_pred_str = 'Current prediction:' + str('{:10.4f}'.format(float(curr_pred)))
                st.markdown(curr_pred_str)
                st.markdown('----')

                params, mse = __optimize_params__(st.session_state['opt_values_df'].iloc[0].values, 
                                            model, 
                                            st.session_state['shap_explainer'], 
                                            target_output,
                                            df_vars_cleaned,
                                            st.session_state['df'])
                st.write('Final params', params)
                st.write('prediction:', model.predict([params]))        
            else:
                curr_pred = model.predict(st.session_state['opt_values_df'].iloc[0:1])
                curr_pred_str = 'Current prediction:' + str('{:10.4f}'.format(float(curr_pred)))
                st.markdown(curr_pred_str)
                st.markdown('----')
                st.write('st.session_state[\'opt_values_df\'].iloc[0].values', st.session_state['opt_values_df'].iloc[0].values)
                st.write('st.session_state[\'opt_values_df\'].iloc[0:1]', st.session_state['opt_values_df'].iloc[0:1])
                #perceber se faz sentido ter aqui um for ou outra cena
                
                params, mse = __optimize_params__(st.session_state['opt_values_df'].iloc[0].values, 
                                            model, 
                                            st.session_state['shap_explainer'], 
                                            target_output,
                                            st.session_state['df_vars_cleaned'],
                                            st.session_state['df'])
                #st.write('Final params', params)
                #st.write('prediction:', model.predict([params])
                st.markdown('---')
                #pred_output_str = '{:10.4f}'.format(model.predict([params]))
                pred_output_str = 'Predicted output: ' + str('{:10.4f}'.format(float(model.predict([params]))))
                st.markdown(pred_output_str)
                table_str = f"""
| Variable | Value |
| ---- | :----: |"""
                mark = []
                for i,v in enumerate(st.session_state['df_vars_cleaned']):
                    #if v[0] == st.session_state['df_vars'][i][0]:              --> meter no json
                    #     st.sess
                    mark.append(f"""
|{v[0]} | {str('{:10.5f}'.format(params[i]))} |""")

                full_table_str = table_str

                for i,v in enumerate(mark):
                    full_table_str = full_table_str + v

                st.markdown(full_table_str)
                st.markdown('---')