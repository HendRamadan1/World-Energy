import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
df = pd.read_csv("/home/skillissue/Summer25/World Energy /data/processed/final_model_ready.csv")
def PlotInfo(country):
    # Filter data for selected country
    df_country = df[df['country_name'] == country]
    
    # Get unique parameters and products
    parameters = sorted(df['parameter'].unique())
    products = sorted(df['product'].unique())
    
    # Create color mapping for products
    colors = px.colors.qualitative.Alphabet[:len(products)]
    color_map = {prod: colors[i] for i, prod in enumerate(products)}
    
    # Create subplots (2 rows x 3 columns)
    fig = make_subplots(
        rows=2, 
        cols=3,
        subplot_titles=parameters,
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Add traces for each parameter-product combination
    for i, param in enumerate(parameters, 1):
        for product in products:
            # Filter data for specific parameter and product
            df_filtered = df_country[
                (df_country['parameter'] == param) & 
                (df_country['product'] == product)
            ].sort_values('date')
            
            # Add trace to subplot
            fig.add_trace(
                go.Scatter(
                    x=df_filtered['date'],
                    y=df_filtered['value'],
                    name=product,
                    legendgroup=product,
                    showlegend=(i == 1),  # Only show legend for first subplot
                    line=dict(color=color_map[product], width=1.5),
                    mode='lines',
                    hovertemplate='Date: %{x|%b %Y}<br>Value: %{y:,.0f} GWh'
                ),
                row=(i-1)//3 + 1,
                col=(i-1)%3 + 1
            )
    
    # Update layout
    fig.update_layout(
        height=900,
        width=1400,
        title_text=f"Energy Production and Consumption Analysis: {country}",
        title_x=0.5,
        legend_title="Products",
        hovermode='x unified'
    )
    
    # Update axis properties
    fig.update_xaxes(title_text='Date', tickformat='%Y', dtick='M12')
    fig.update_yaxes(title_text='GWh')
    
    # Add annotations for empty subplots
    for i in range(len(parameters) + 1, 7):
        fig.add_annotation(
            row=(i-1)//3 + 1,
            col=(i-1)%3 + 1,
            text="No Data Available",
            showarrow=False,
            font=dict(size=15)
            )
    
    return fig