import streamlit as st

# Define the pages
HomePage = st.Page("pages/main.py", title="Main Page", icon="🎈")
TransformerModel = st.Page("pages/transformer_model.py", title="Transformer Model", icon="❄️")
Eda = st.Page("pages/eda.py", title="EDA", icon="🎉")
FeatureEngineering = st.Page("pages/feature_engineering.py", title="Feature Engineering", icon="🎉")
pg = st.navigation([HomePage, TransformerModel, Eda, FeatureEngineering])
# # Set up horizontal navigation using a dictionary
# pg = st.navigation({
#     "Main": [HomePage],
#     "Model": [TransformerModel],
#     "Analysis": [Eda],
# })

# Run the selected page
pg.run()