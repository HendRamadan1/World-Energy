# visualizers.py

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load data once at startup
df = pd.read_csv("data/processed/final_model_ready.csv")

class Plotter:
    def plotInfo(self, countryName: str):
        """Generate multi-subplot time series visualization for a given country."""
        
        selected_country = df[df['country_name'] == countryName]

        if selected_country.empty:
            return None

        parameters = sorted(selected_country['parameter'].unique())
        all_products = sorted(df['product'].unique())

        colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
        ]
        color_discrete_map = dict(zip(all_products, colors))

        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=parameters,
            vertical_spacing=0.12,
            horizontal_spacing=0.05
        )

        for i, param in enumerate(parameters):
            row = (i // 2) + 1
            col = (i % 2) + 1

            param_data = selected_country[selected_country['parameter'] == param]
            pivot = param_data.pivot_table(
                index='date',
                columns='product',
                values='value',
                aggfunc='sum'
            ).reset_index()
            melted = pivot.melt(id_vars='date', value_name='value', var_name='product')

            full_dates = pd.date_range(
                start=selected_country['date'].min(),
                end=selected_country['date'].max(),
                freq='MS'
            )

            for product in all_products:
                product_data = melted[melted['product'] == product].copy()
                product_data = product_data.set_index('date').reindex(full_dates).fillna(0).reset_index()
                product_data['product'] = product

                fig.add_trace(
                    go.Scatter(
                        x=product_data['index'],
                        y=product_data['value'],
                        name=product,
                        mode='lines',
                        line=dict(width=2),
                        marker_color=color_discrete_map[product],
                        showlegend=(i == 0)
                    ),
                    row=row,
                    col=col
                )

        fig.update_layout(
            title_text=f'{countryName} Electricity Data (2010-2023)',
            height=1200,
            width=1400,
            legend_title_text='Product Types',
            hovermode='x unified'
        )

        for i in range(len(parameters)):
            row = (i // 2) + 1
            col = (i % 2) + 1
            fig.update_xaxes(title_text='Date', row=row, col=col)
            fig.update_yaxes(title_text='Value (GWh)', row=row, col=col)

        for idx, annotation in enumerate(fig.layout.annotations):
            if idx < len(parameters):
                row = 3 - (idx // 2)
                y_pos = 1.0 - (row - 1) * (0.33 + 0.04)
                annotation.update(y=y_pos, yanchor='bottom')

        return fig