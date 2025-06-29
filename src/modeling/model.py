import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class SeasonalLSTM(nn.Module):
    """LSTM model for seasonal time series forecasting.

    Args:
        input_size (int): Number of input features.
        hidden_size (int): Number of hidden units in the LSTM.
        output_size (int): Number of output features (predicted seasons).
        num_layers (int): Number of LSTM layers. Defaults to 2.
        dropout (float): Dropout rate for LSTM layers. Defaults to 0.2.
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.2):
        super(SeasonalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Forward pass through the LSTM model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def create_sequences(data, country, parameter, product, input_years=3, predict_seasons=1):
    """Create input sequences and targets for the SeasonalLSTM model.

    Args:
        data (pd.DataFrame): The complete dataset.
        country (str): Country to filter data.
        parameter (str): Parameter to filter data.
        product (str): Product to filter data.
        input_years (int): Number of years for input sequence. Defaults to 3.
        predict_seasons (int): Number of seasons to predict. Defaults to 1.

    Returns:
        tuple: (X, y, preprocessor, target_scaler)
            - X (np.array): Input sequences.
            - y (np.array): Target values.
            - preprocessor (ColumnTransformer): Preprocessor for features.
            - target_scaler (MinMaxScaler): Scaler for target values.
    """
    country_data = data[
        (data['country_name'] == country) &
        (data['parameter'] == parameter) &
        (data['product'] == product)
    ].sort_values('date').copy()

    country_data['rolling_avg'] = country_data['value'].rolling(window=3, min_periods=1).mean()
    country_data = country_data.dropna()

    hemisphere = country_data['hemisphere'].iloc[0]
    seasons = {
        'Northern': {
            'Winter': [12, 1, 2], 'Spring': [3, 4, 5], 'Summer': [6, 7, 8], 'Fall': [9, 10, 11]
        },
        'Southern': {
            'Summer': [12, 1, 2], 'Fall': [3, 4, 5], 'Winter': [6, 7, 8], 'Spring': [9, 10, 11]
        }
    }[hemisphere]

    month_to_season = {month: season for season, months in seasons.items() for month in months}
    country_data['season'] = country_data['month'].map(month_to_season)

    seasonal_avg = country_data.groupby(['year', 'season']).agg({
        'rolling_avg': 'mean', 'major_production_month': 'last', 'production_season': 'last',
        'peak_ratio': 'last', 'peak_consistency': 'last', 'hemisphere': 'first'
    }).reset_index()

    season_order = list(seasons.keys())
    seasonal_avg['season'] = pd.Categorical(seasonal_avg['season'], categories=season_order, ordered=True)
    seasonal_avg = seasonal_avg.sort_values(['year', 'season'])

    features = seasonal_avg[['rolling_avg', 'major_production_month', 'peak_ratio', 'peak_consistency', 'production_season', 'hemisphere']]
    target = seasonal_avg['rolling_avg']

    preprocessor = ColumnTransformer([
        ('num', MinMaxScaler(), ['rolling_avg', 'major_production_month', 'peak_ratio', 'peak_consistency']),
        ('cat', OneHotEncoder(), ['production_season', 'hemisphere'])
    ])

    processed_features = preprocessor.fit_transform(features)
    target_scaler = MinMaxScaler()
    target_scaled = target_scaler.fit_transform(target.values.reshape(-1, 1))

    input_seasons = input_years * 4
    total_seasons = len(seasonal_avg)
    X, y = [], []

    for i in range(total_seasons - input_seasons - predict_seasons + 1):
        X.append(processed_features[i:i+input_seasons])
        y.append(target_scaled[i+input_seasons:i+input_seasons+predict_seasons].flatten())

    return np.array(X), np.array(y), preprocessor, target_scaler