import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(page_title="My App", layout="wide")


selected = option_menu(
    None,
    ["Home", "EDA", "Model", "Feature Engineering"],
    icons=["house", "bar-chart-line", "robot", "sliders"],
    orientation="horizontal",
    key="top_menu"
)


if selected == "Home":
    HomePage = st.Page("pages/main.py", title="Main Page", icon="🏠")
    pg = st.navigation([HomePage])
    pg.run()
    
    

elif selected == "EDA":
    Eda = st.Page("pages/eda.py", title="EDA", icon="📊")
    pg = st.navigation([Eda])
    pg.run()

elif selected == "Model":
    TransformerModel = st.Page("pages/transformer_model.py", title="Transformer Model", icon="🤖")
    pg = st.navigation([TransformerModel])
    pg.run()
    
elif selected == "Feature Engineering":
    FeatureEngineering = st.Page("pages/feature_engineering.py", title="Feature Engineering", icon="🛠")
    pg = st.navigation([FeatureEngineering])
    pg.run()
    
