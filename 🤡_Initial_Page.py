import pickle
import streamlit as st
from pathlib import Path
import json
import os
import shutil
import pandas as pd

st.set_page_config(
     page_title="Argentina Optimization Toolkit",
     page_icon="🤡",
     initial_sidebar_state="auto"
 )



def st_directory_picker(initial_path=Path()):

    st.markdown("#### Choose a project directory")

    if "path" not in st.session_state:
        st.session_state.path = initial_path.absolute()

    st.text_input("Selected directory:", st.session_state.path)

    _, col1, col2, col3, _ = st.columns([3, 1, 3, 1, 3])

    with col1:
        st.markdown("#")
        if st.button("⬅️") and "path" in st.session_state:
            st.session_state.path = st.session_state.path.parent
            st.experimental_rerun()

    with col2:
        subdirectroies = [
            f.stem
            for f in st.session_state.path.iterdir()
            if f.is_dir()
            and (not f.stem.startswith(".") and not f.stem.startswith("__"))
        ]
        if subdirectroies:
            st.session_state.new_dir = st.selectbox(
                "Subdirectories", sorted(subdirectroies)
            )
        else:
            st.markdown("#")
            st.markdown(
                "<font color='#FF0000'>No subdir</font>", unsafe_allow_html=True
            )

    with col3:
        if subdirectroies:
            st.markdown("#")
            if st.button("➡️") and "path" in st.session_state:
                st.session_state.path = Path(
                    st.session_state.path, st.session_state.new_dir
                )
                st.experimental_rerun()

    return st.session_state.path 

st.markdown('# Optimization Toolkit')

if 'root_dir' not in st.session_state:

    root_dir = st_directory_picker()
    if st.button('Select'):
        params_check_str = str(root_dir) + '\\params.json'
        if not(os.path.exists(params_check_str)):  
            st.session_state['root_dir'] = root_dir
            
            curr_dir = curr_dir = os.getcwd()
            json_file = str(st.session_state['root_dir']) + '\\params.json'
            #https://www.geeksforgeeks.org/read-json-file-using-python/
            if not(os.path.isfile(json_file)):
                    template_file = curr_dir + '\\template.json'
                    shutil.copy(template_file, json_file)
        else:
            st.session_state['root_dir'] = root_dir

            params_str = str(st.session_state['root_dir']) + '\\params.json'
            with open(params_str, 'r+') as file:
                data = json.load(file)
            
            st.session_state['df_vars'] = data['df_vars'][0]
            st.session_state['output_var'] = data['output_var']
            st.session_state['feature_names'] = data['feature_names']
            st.session_state['model_name'] = data['model_name']
            st.session_state['t_t_ratio'] = data['t_t_ratio']

            #unpickle o modelo, meu amigo
            model_file = str(st.session_state['root_dir']) + '\\model.sav'
            st.session_state['model'] = pickle.load(open(model_file, 'rb'))
            
            df_str = str(st.session_state['root_dir']) + '\\df.pkl'
            df_no_output_str = str(st.session_state['root_dir']) + '\\df_no_output.pkl'
            st.session_state['df'] = pd.read_pickle(df_str)
            st.session_state['df_no_output'] = pd.read_pickle(df_no_output_str)
else:
    st.markdown('## Parabéns')
    st.markdown('To select a different folder, reload the app')
    