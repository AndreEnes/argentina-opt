import streamlit as st
from sklearn.model_selection import train_test_split
from opt_model import __train_model__

st.set_page_config(
     page_title="Argentina Optimization Toolkit",
     page_icon="🤡",
     initial_sidebar_state="auto"
 )

st.session_state['page_change'] = True

st.markdown('# Current variable state')

if 'df_vars' in st.session_state:
    st.header('Output')

    if 'output_var' not in st.session_state:
        st.write('No output chosen')
    else:
        st.markdown(str(st.session_state['output_var']))
    st.markdown('---')
    st.markdown('### Current variable state')
    table_str = f"""
| Variable | Range | Type | Minimal increment |
| :---- | :----: | :----: | :----: |"""
    increment = 'None'
    mark = []

    for i,v in enumerate(st.session_state['df_vars']):
        if v[1] == None:
            limits = 'Undefined'
        else:
            limits = str(v[1])
        if v[2] == None:
            v_type = 'Continuous'
        else:
            v_type = 'Discrete'

            if v[3] == 0:
                increment = '1'
            else:
                increment = str(v[3])
                
        mark.append(f"""
| {v[0]} | {limits} | {v_type} | {increment} |""")

        increment = 'None'

    full_table_str = table_str 
    for i,v in enumerate(mark):
        full_table_str = full_table_str + v
    
    st.markdown(full_table_str)
    st.markdown('---')

    if 'model' in st.session_state:
        st.header('Model')
        st.markdown('## XGBoost Regression')

        table_str = f"""
|**R2** | **MAE** | **MSE** |
| :----: | :----: | :----: |"""

        st.markdown('#### Training data')
        train_str = f"""
|{str('{:10.4f}'.format(st.session_state['model_error'][0]))} | {str('{:10.4f}'.format(st.session_state['model_error'][1]))} | {str('{:10.4f}'.format(st.session_state['model_error'][2]))} |"""
        full_table_str = table_str + train_str
        st.markdown(full_table_str)
        
        st.markdown('#### Testing data')
        test_str = f"""
|{str('{:10.4f}'.format(st.session_state['model_error'][3]))} | {str('{:10.4f}'.format(st.session_state['model_error'][4]))} | {str('{:10.4f}'.format(st.session_state['model_error'][5]))} |"""
        full_table_str = table_str + test_str
        st.markdown(full_table_str)


        st.markdown('-----')
else:
    st.write('Variables do not exist at this moment')