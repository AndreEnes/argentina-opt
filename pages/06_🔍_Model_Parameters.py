import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

shap.initjs()

def __model_params__(model, x_train, n_samples):
    #https://shap.readthedocs.io/en/latest/generated/shap.explainers.Sampling.html
    explainer = shap.TreeExplainer(model)
    #get Shapley values --> model params
    shap_values = explainer.shap_values(x_train[:n_samples])
    #print butterfly plot
    shap.summary_plot(shap_values, x_train[:n_samples], show=False)
    plt.savefig('model_params_summary.png', bbox_inches='tight', dpi=100)
    figure_summary = 'model_params_summary.png'

    return figure_summary, shap_values, explainer


st.set_page_config(
     page_title="Argentina Optimization Toolkit",
     page_icon="🔍",
     initial_sidebar_state="auto"
 )

st.title('Model parameters')

if 'model' in st.session_state:
    model = st.session_state['model']

    if 'x_train' in st.session_state:
        x_train = st.session_state['x_train']
        x_test = st.session_state['x_test']
        y_train = st.session_state['y_train']
        y_test = st.session_state['y_test']
    else:
        df_no_output = st.session_state['df_no_output']
        df = st.session_state['df']
        output_var = st.session_state['output_var']
        t_t_ratio = st.session_state['t_t_ratio'][0]
        x_train, x_test, y_train, y_test = train_test_split(df_no_output, 
                                                        df[output_var], 
                                                        test_size=float(t_t_ratio), 
                                                        random_state=42)

    x_tot = pd.concat([x_train, x_test])
    y_tot = pd.concat([y_train, y_test])

    caption = str(st.session_state['model_name']) + ' - feature influence on the output'

    if 'model_params_image' not in st.session_state:
        image, shap_values, explainer = __model_params__(model, x_tot, x_tot.shape[0])
        st.image(image, caption=caption)
        st.session_state['model_params_image'] = image
        st.session_state['shap_values'] = shap_values
        st.session_state['shap_explainer'] = explainer
    else:
        st.image(st.session_state['model_params_image'], caption=caption)
        if st.button('Rerun'):
            image, shap_values, explainer = __model_params__(model, x_tot, x_tot.shape[0])
            st.image(image, caption=caption)
            st.session_state['model_params_image'] = image
            st.session_state['shap_values'] = shap_values
            st.session_state['shap_explainer'] = explainer