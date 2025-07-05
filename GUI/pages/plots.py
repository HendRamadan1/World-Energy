from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
from eda import df

class Plotter():
    def plotInfo(countryName:str):
        """choose a country from the following: ['Argentina', 'Australia', 'Austria', 'Belgium', 'Brazil',
        'Bulgaria', 'Canada', 'Chile', 'China', 'Colombia', 'Costa Rica',
        'Croatia', 'Cyprus', 'Czech Republic', 'Denmark', 'Estonia',
        'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Iceland',
        'India', 'Ireland', 'Italy', 'Japan', 'Korea', 'Latvia',
        'Lithuania', 'Luxembourg', 'Malta', 'Mexico', 'Netherlands',
        'New Zealand', 'North Macedonia', 'Norway', 'Peru', 'Poland',
        'Portugal', 'Serbia', 'Slovak Republic', 'Slovenia', 'Spain',
        'Sweden', 'Switzerland', 'Turkey', 'United Kingdom',
        'United States']"""
        
        selected_country = df[df['country_name'] == countryName]

        # Get unique parameters and products
        parameters = sorted(selected_country['parameter'].unique())
        all_products = sorted(df['product'].unique())

        # Create consistent color mapping for all products
        # Use 15 distinct colors from qualitative palettes (Plotly, D3, Alphabet)
        colors = [
            '#1f77b4',  # Plotly: Blue
            '#ff7f0e',  # Plotly: Orange
            '#2ca02c',  # Plotly: Green
            '#d62728',  # Plotly: Red
            '#9467bd',  # Plotly: Purple
            '#8c564b',  # Plotly: Brown
            '#e377c2',  # Plotly: Pink
            '#7f7f7f',  # Plotly: Gray
            '#bcbd22',  # Plotly: Olive
            '#17becf',  # Plotly: Cyan
            '#aec7e8',  # D3: Light Blue
            '#ffbb78',  # D3: Light Orange
            '#98df8a',  # D3: Light Green
            '#ff9896',  # D3: Light Red
            '#c5b0d5'   # D3: Light Purple
        ]
        color_discrete_map = dict(zip(all_products, colors))

        # Create subplot grid (3 rows, 2 columns)
        fig = make_subplots(
            rows=3, 
            cols=2,
            subplot_titles=parameters,
            vertical_spacing=0.12,
            horizontal_spacing=0.05
        )

        # Process each parameter
        for i, param in enumerate(parameters):
            # Get row and column position
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            # Filter data for current parameter
            param_data = selected_country[selected_country['parameter'] == param]
            
            # Create pivot table to ensure complete date-product combinations
            pivot = param_data.pivot_table(
                index='date',
                columns='product',
                values='value',
                aggfunc='sum'
            ).reset_index()
            
            # Melt to long format for Plotly
            melted = pivot.melt(id_vars='date', value_name='value', var_name='product')
            
            # Add traces for each product
            for product in all_products:
                product_data = melted[melted['product'] == product]
                
                # Fill missing dates with 0
                full_dates = pd.date_range(
                    start=selected_country['date'].min(),
                    end=selected_country['date'].max(),
                    freq='MS'
                )
                product_data = product_data.set_index('date').reindex(full_dates).fillna(0).reset_index()
                product_data['product'] = product
                
                # Add trace to subplot
                fig.add_trace(
                    go.Scatter(
                        x=product_data['index'],
                        y=product_data['value'],
                        name=product,
                        mode='lines',
                        line=dict(width=2),
                        marker_color=color_discrete_map[product],
                        showlegend=(i == 0)  # Only show legend for first subplot
                    ),
                    row=row,
                    col=col
                )

        # Update layout
        fig.update_layout(
            title_text=f'{countryName} Electricity Data (2010-2023)',
            height=1200,
            width=1400,
            legend_title_text='Product Types',
            hovermode='x unified'
        )

        # Format axes
        for i in range(1, 7):
            fig.update_xaxes(title_text='Date', row=(i+1)//2 if i%2==1 else (i)//2, col=1 if i%2==1 else 2)
            fig.update_yaxes(title_text='Value (GWh)', row=(i+1)//2, col=1 if i%2==1 else 2)

        # Improve subplot title positioning
        for idx, annotation in enumerate(fig['layout']['annotations']):
            # Calculate y position based on subplot row (3 rows, top row starts at y=1)
            row = 3 - (idx // 2)  # Row 1 (top) -> y~1, Row 2 -> y~0.62, Row 3 -> y~0.24
            y_pos = 1.0 - (row - 1) * (0.33 + 0.04)  # Adjust for increased vertical spacing
            annotation.update(y=y_pos, yanchor='bottom')

        fig.show()

obj = Plotter()
obj.plotInfo()