import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv(r"../data/processed/final_model_ready.csv")
st.markdown("## Exploratory  Data Analysis")
st.markdown("""
       ### 🔍 Steps of (EDA)""")

with st.expander("**Understanding the Data Structure**"):
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
    


with st.expander("**Summary of statistics**"):
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
    - 🔸 **Max value (865,976)** is much larger than 75% of the data → strong indication of **outliers** but not really, this is because when grouped by countries there are no outliers, funny how someone could easily lie with statistics sometimes.
    - 🔸 Most values lie between **41 and 2,600**, but some extreme values inflate the average.
""")
    
    
     
with st.expander("**Check and handle Duplicates**"):
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
  
  
    
with st.expander("**Check and handle Missing values**"):
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
    # visuals here     



with st.expander("**Insights**"):
    st.markdown("""

     """)




def PlotInfo(country):
    # Filter data for selected country
    df_country = df[df['country_name'] == country]
    
    # Get unique parameters and products
    parameters = sorted(df['parameter'].unique())
    products = sorted(df['product'].unique())
    
    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    fig.suptitle(f'Energy Production and Consumption Analysis: {country}', 
                 fontsize=24, y=0.98)
    
    # Create colormap for products (15 distinct colors)
    colors = plt.cm.tab20(np.linspace(0, 1, len(products)))
    product_colors = dict(zip(products, colors))
    
    # Create a dictionary to store legend handles
    legend_handles = {}
    
    # Plot each parameter in its own subplot
    for ax, param in zip(axes.flat, parameters):
        # Filter data for current parameter
        param_data = df_country[df_country['parameter'] == param]
        
        # Plot each product
        for product in products:
            product_data = param_data[param_data['product'] == product]
            if not product_data.empty:
                # Plot the line
                line = ax.plot(
                    product_data['date'], 
                    product_data['value'], 
                    color=product_colors[product],
                    linewidth=1.5,
                    alpha=0.9
                )
                # Store handle for legend if not already stored
                if product not in legend_handles:
                    legend_handles[product] = line[0]
        
        # Subplot formatting
        ax.set_title(param, fontsize=16, pad=12)
        ax.set_ylabel('GWh', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='x', rotation=45)
        ax.xaxis.set_major_locator(plt.MaxNLocator(8))  # Limit x-axis ticks
    
    # Create comprehensive legend
    fig.legend(
        handles=legend_handles.values(),
        labels=legend_handles.keys(),
        loc='lower center',
        ncol=3,
        fontsize=12,
        bbox_to_anchor=(0.5, -0.05),
        frameon=True,
        title="Energy Products",
        title_fontsize=14
    )
    
    # Adjust layout to make room for legend
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.subplots_adjust(bottom=0.15, hspace=0.3, wspace=0.2)
    
    return fig

# Streamlit app integration
st.title("Country Info plot")
selected_country = st.selectbox(
    "Select Country:",
    sorted(df['country_name'].unique()),
    index=0
)

if st.button('run'):
    fig = PlotInfo(selected_country)
    st.pyplot(fig)
    


def SeasonalDiff(parameter: str):
    # Make sure we have a copy of the dataframe
    param_data = df[df['parameter'] == parameter].copy()
    
    # Convert date column to datetime
    param_data['date'] = pd.to_datetime(param_data['date'], errors='coerce')
    param_data = param_data.dropna(subset=['date'])
    
    # Get all possible products (ensure we have all 15)
    all_products = sorted(df['product'].unique())
    
    # Get top 8 countries by average value
    top_countries = param_data.groupby('country_name')['value'].mean().nlargest(8).index
    filtered_data = param_data[param_data['country_name'].isin(top_countries)].copy()
    
    # Season mapping
    season_map = {
        1: 'Winter', 2: 'Winter', 3: 'Spring', 
        4: 'Spring', 5: 'Spring', 6: 'Summer',
        7: 'Summer', 8: 'Summer', 9: 'Fall',
        10: 'Fall', 11: 'Fall', 12: 'Winter'
    }
    filtered_data['season'] = filtered_data['date'].dt.month.map(season_map)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Winter', 'Spring', 'Summer', 'Fall'],
        specs=[[{'type': 'bar'}, {'type': 'bar'}], 
              [{'type': 'bar'}, {'type': 'bar'}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Color palette for all 15 products
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
    ]
    color_map = dict(zip(all_products, colors[:len(all_products)]))
    
    # Process each season
    seasons = ['Winter', 'Spring', 'Summer', 'Fall']
    for i, season in enumerate(seasons):
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        season_data = filtered_data[filtered_data['season'] == season]
        
        # Create pivot with all products (even if missing)
        pivot = season_data.pivot_table(
            index='country_name',
            columns='product',
            values='value',
            aggfunc='mean'
        )
        
        # Ensure all 15 products are represented
        for product in all_products:
            if product not in pivot.columns:
                pivot[product] = 0  # Add missing products with 0 values
        
        # Reorder columns to match all_products
        pivot = pivot[all_products]
        
        # Add traces for each product
        for product in all_products:
            fig.add_trace(
                go.Bar(
                    x=pivot.index,
                    y=pivot[product],
                    name=product,
                    marker_color=color_map[product],
                    showlegend=(i == 0),  # Show legend only once
                    hoverinfo='y+name'
                ),
                row=row, col=col
            )
    
    # Update layout
    fig.update_layout(
        title_text=f'Seasonal Comparison of {parameter} (Top 8 Countries)',
        height=900,
        barmode='stack',
        margin=dict(l=50, r=50, b=100, t=100, pad=4),
        legend_title_text='Energy Products'
    )
    
    # Update axis labels
    for i in range(1, 5):
        fig.update_yaxes(title_text="Value (GWh)", row=(i+1)//2, col=1 if i%2==1 else 2)
    
    return fig

# Streamlit app
st.markdown('## Seasonal Energy Comparison')

# Parameter selection
parameter = st.selectbox(
    'Select Parameter:',
    sorted(df['parameter'].unique())
)

# Generate plot
fig = SeasonalDiff(parameter)
st.plotly_chart(fig, use_container_width=True)