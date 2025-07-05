import streamlit as st


st.markdown("## World Energy Prediction")
st.image("https://storage.googleapis.com/kaggle-datasets-images/4633877/7892412/5016a8678efd1decad688d6966a5c0e4/dataset-cover.png?t=2024-03-20-06-59-09")
st.markdown("This project utilizes the [world electricity dataset](https://www.kaggle.com/datasets/sazidthe1/global-electricity-production) to predict electricity production till the year 2025 using a Transformer model, this project used smart data cleaning and grouping, feature engineering, and eye pleasing visuals")
st.markdown("### Exploratory data analysis (EDA)")
st.markdown("EDA plays a crucial role in understanding the underlying structure of the dataset. Through visualizations and statistical summaries, we identified trends, missing data, outliers, and correlations between variables such as country, year, energy source, and production levels. This step guided subsequent preprocessing and modeling decisions.")
st.markdown("### Feature Engineering")
st.markdown("To enhance model performance, we engineered new features from the raw data, including temporal features (e.g., year, decade), categorical embeddings for country and energy type, and normalized production values. We also applied smart grouping and aggregation strategies to reduce noise and improve generalization. These transformations enabled the model to better understand and predict complex energy production patterns.")
st.markdown("### Transformers")
st.markdown("The Transformer model, originally designed for natural language processing, has been adapted for time-series forecasting due to its ability to capture long-range dependencies and patterns. In this project, a Transformer-based architecture is used to predict future electricity production by learning from historical data trends across countries and energy sources.")
st.markdown("## Contributors: ")
st.markdown("- _Yousef Fawzi_ \n - _Hend Ramadan_")
st.sidebar.markdown("# Contact us! \n - ### [Hend Ramadan](mailto:hendtalba@gmail.com) \n - ### [Yousef Fawzi](mailto:losif.ai.2050@gmail.com) \n ### [github repo](https://github.com/Losif01/World-Energy)")
