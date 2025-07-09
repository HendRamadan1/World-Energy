
---
# 🌍 World Energy Prediction project (using transformers)

A data-driven project that explores global electricity production and predicts future trends using advanced modeling techniques.

This project leverages the [World Electricity Dataset](https://www.kaggle.com/datasets/sazidthe1/global-electricity-production) to analyze historical energy production and forecast values up to 2025 using a Transformer-based model.

---

## 🧩 Project Overview

This interactive web application provides:

- **Exploratory Data Analysis (EDA)**: Visualize historical trends in energy production across countries.
- **Feature Engineering**: Add economic, seasonal, and geographic context to enhance predictive modeling.
- **Transformer Model**: Predict future electricity production using deep learning with attention mechanisms.
- **Interactive Visualizations**: Understand patterns and anomalies through dynamic plots.

The app is built using **Streamlit**, allowing seamless navigation between different modules via a top menu bar.

---

## 🛠️ Features

### 1. Exploratory Data Analysis (EDA)
- Interactive time-series plots by country and energy source.
- Seasonal analysis of production levels.
- Economic indicators such as GDP and GDP per capita over time.

### 2. Feature Engineering
- Adds hemisphere classification, daylight hours, and inferred nuclear plant outages.
- Enriches dataset with external GDP and GDP per capita data from the World Bank API.

### 3. Transformer-Based Forecasting
- Adapts natural language processing architecture for time-series forecasting.
- Learns complex temporal dependencies to predict electricity production trends.

---

## 📦 GUI File Structure

```
GUI/
│
├── app.py                      # Main Streamlit app with navigation
├── pages/
│   ├── main.py                 # Home page with project overview
│   ├── eda.py                  # EDA module with visualizations
│   ├── feature_engineering.py  # Feature engineering explanations and logic
│   └── transformer_model.py    # Transformer model implementation
│
├── data/
│   └── processed/
│       └── final_model_ready.csv  # Final cleaned and engineered dataset
│
└── README.md                   # This file
```

---

## 🚀 Getting Started

### Prerequisites

Make sure you have the following installed:

- Python 3.8+
- `streamlit`
- `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`
- `torch` for Transformer model
- `pycountry`, `requests`, `astral` for feature engineering

You can install them using pip:

```bash
pip install -r requirements.txt
```

### Running the App

From the GUI directory:

```bash
streamlit run app.py
```

Navigate through the app using the horizontal menu at the top.

---

## 🧑‍💻 Contributors

- [Yousef Fawzi](mailto:losif.ai.2050@gmail.com)
- [Hend Ramadan](mailto:hendtalba@gmail.com)

🔗 [GitHub Repository](https://github.com/Losif01/World-Energy)

---

## 📬 Contact

For questions or contributions, feel free to reach out:

- Yousef Fawzi: [losif.ai.2050@gmail.com](mailto:losif.ai.2050@gmail.com)
- Hend Ramadan: [hendtalba@gmail.com](mailto:hendtalba@gmail.com)

---

## 📄 License

MIT License – see the [LICENSE](LICENSE) file for details.

---