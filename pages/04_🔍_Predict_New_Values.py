import streamlit as st
import pandas as pd

def __pred_new_value__(values, model):
    prediction = model.predict(values.values)
    return prediction

st.set_page_config(
     page_title="Argentina Optimization Toolkit",
     page_icon="🔍",
     initial_sidebar_state="auto"
 )

st.session_state['page_change'] = True

st.title('Predict new values')

if 'df' not in st.session_state:
    st.write('Configure model before accessing this page')
elif 'model' not in st.session_state:
    st.write('Choose a model before accessing this page')
else:
    new_values_file = st.file_uploader('Choose a .csv file containing the new values')

    if new_values_file is not None:
        df_new = pd.read_csv(new_values_file)

        model = st.session_state['model']
        pred = []
        st.table(df_new)
        for i in range(len(df_new)):
            pred = __pred_new_value__(df_new.iloc[i:i+1], model)[0]
            pred_str = '**Prediction**: ' + str('{:10.4f}'.format(__pred_new_value__(df_new.iloc[i:i+1], model)[0]))
            st.markdown(pred_str)       