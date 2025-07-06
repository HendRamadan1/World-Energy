import streamlit as st
import pandas as pd
df = pd.read_csv(r"/home/skillissue/Summer25/World Energy /data/processed/final_model_ready.csv")
st.markdown("## Exploratory  Data Analysis")
st.markdown("""
       ### 🔍 Steps of (EDA)""")

with st.expander(" 1️⃣ **Understanding the Data Structure**"):
     st.markdown("### subset from Data :")
     st.dataframe({
        "country_name": ["Australia", "Australia", "Australia","Australia"],
        "date": ["2023-01-01", "2023-01-01", "2023-01-01","2023-01-01"],
        "parameter": ["Net Electricity Production", "Net Electricity Production", "Net Electricity Production","Net Electricity Production"],
        "product": ["Electricity", "Total Combustible Fuels", "Coal, Peat and Manufactured Gases","Oil and Petroleum Products"],
        "value": [22646.1901, 13397.9356, 9768.5223,289.5415],
        "unit": ["GWh", "GWh", "GWh","GWh"]
    })
     st.markdown("""
                                         
   -  ```python
         data.info()
         
      ```
      Example Output :
      
      ```
      <class 'pandas.core.frame.DataFrame'>
        RangeIndex: 121074 entries, 0 to 121073
        Data columns (total 6 columns):
        #   Column        Non-Null Count   Dtype  
        ---  ------        --------------   -----  
        0   country_name  121074 non-null  object 
        1   date          121074 non-null  object 
        2   parameter     121074 non-null  object 
        3   product       121074 non-null  object 
        4   value         121060 non-null  float64
        5   unit          121074 non-null  object 
        dtypes: float64(1), object(5)
        memory usage: 5.5+ MB
      ```
      
      ```python
        print(df.shape) 
      ```
        shape Output :
        
      ```
        (121074, 6)
      ```
      
      ```python
        print(df.colums) 
      ```
      
      columns Output:
      
      ```
       Index(['country_name', 'date', 'parameter', 'product', 'value', 'unit'], dtype='object')  
      ```
      
      ```python
        print(df.dtypes) 
      ```
      
       Output:
      
      ```
         country_name       object
        date             object
        parameter        object
        product          object
        value           float64
        unit             object 
      ```     
     """)
    


with st.expander(" 2️⃣ **Summary of statistics**"):
    st.markdown("""
                                         
   -  ```python
         data.descirbe()
         
      ```
      Output :
      
      ```
     
                 value    
        ---      --------------   
        count    121060.0 
        mean     6925.081489784404          
        std      34224.45546033089     
        min      0.0       
        25%      41.199                  
        50%      470.419 
        75%      2629.71275
        max      865976.4828
      ```  
     """)
    st.markdown("""
    **Insights:**

    - 🔸 There's a large gap between the **mean (6925)** and **median (470)** →  data is  **right-skewed**.
    - 🔸 High **standard deviation** (34224) suggests significant variation.
    - 🔸 **Max value (865,976)** is much larger than 75% of the data → strong indication of **outliers**.
    - 🔸 Most values lie between **41 and 2,600**, but some extreme values inflate the average.
""")
    
    
     
with st.expander(" 3️⃣ **Check and handle Duplicates**"):
    st.markdown("""
                                         
   -  ```python
        df.duplicated().sum()
         
      ```
      Output :
      
      ```
      np.int64(0)
      ``` 
      
      **This meaning there  is no duplicated values in the data** 
     """)
  
  
    
with st.expander(" 4️⃣ **Check and handle Missing values**"):
    st.markdown("""
    ### check missing value                                   
    ```python
    def getMissingValues(df):
        missing_values = df.isnull().sum().sort_values(ascending=False)
        missing_values = missing_values[missing_values > 0]
        missing_values = missing_values / len(df)
        return [missing_values], missing_values.__len__()
    print(getMissingValues(df)) 
    ```
      Output :
      
      ```
      ([value    0.000116
        dtype: float64], 1)
      ```
      ### Handle missing value
      From the output, we observe that :
      Only the **value** column has missing entries
      the missing precentage is approximate 0.01%  witch it very small 
      so missing values have be handled by droping the affected rows
      
      
      ```python
      df = df.dropna(subset=['value'])
      ```
      
      Output:
      
      ```
      ([Series([], dtype: float64)], 0)
      ```
     """)
    
    

with st.expander(" 5️⃣  **visulazation Data**"):
    
    st.markdown("""
    **Detect missing values**
     """)
    st.image('../plots/output.png',caption="Detect missing values with Bar plot")


    
with st.expander(" 6️⃣ **Insights**"):
    st.markdown("""
                                         
                                    welcome yousef in  our project 
     """)
    
