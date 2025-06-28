import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle
import os

class SeasonalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(SeasonalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def create_sequences(data, country, parameter, product, input_years=3, predict_seasons=1):
    country_data = data[
        (data['country_name'] == country) &
        (data['parameter'] == parameter) &
        (data['product'] == product)
    ].sort_values('date').copy()
    
    # Calculate 3-month rolling average
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
    
    features = seasonal_avg[['rolling_avg', 'major_production_month', 'peak_ratio', 'peak_consistency']]
    target = seasonal_avg['rolling_avg']
    
    preprocessor = ColumnTransformer([
        ('num', MinMaxScaler(), ['rolling_avg', 'major_production_month', 'peak_ratio', 'peak_consistency']),
        ('cat', OneHotEncoder(), ['production_season', 'hemisphere'])
    ])
    
    processed_features = preprocessor.fit_transform(seasonal_avg)
    target_scaler = MinMaxScaler()
    target_scaled = target_scaler.fit_transform(target.values.reshape(-1, 1))
    
    input_seasons = input_years * 4
    total_seasons = len(seasonal_avg)
    X, y = [], []
    
    for i in range(total_seasons - input_seasons - predict_seasons + 1):
        X.append(processed_features[i:i+input_seasons])
        y.append(target_scaled[i+input_seasons:i+input_seasons+predict_seasons].flatten())
    
    return np.array(X), np.array(y), preprocessor, target_scaler

def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        criterion = nn.MSELoss()
        loss = criterion(outputs, y_test)
    return loss.item()

if __name__ == "__main__":
    # Load the dataset
    data_path = r"/home/skillissue/Summer25/World Energy /data/processed/model_ready.csv"
    data = pd.read_csv(data_path)
    
    # Prepare test data
    X, y, feature_scaler, target_scaler = create_sequences(
        data, 'Germany', 'Net Electricity Production', 'Electricity'
    )
    
    # Split into test set (same split as in train.py)
    split_idx = int(len(X) * 0.8)
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    
    # Convert to PyTorch tensors
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
    
    # Load the trained model
    model = SeasonalLSTM(
        input_size=X_test.shape[2],
        hidden_size=128,  # Must match train.py
        output_size=y_test.shape[1],
        num_layers=2      # Must match train.py
    )
    model_path = "/home/skillissue/Summer25/World Energy /models/saved_models/seasonal_lstm_model.pth"
    model.load_state_dict(torch.load(model_path))
    
    # Evaluate the model
    test_loss = evaluate_model(model, X_test, y_test)
    print(f"Test Loss: {test_loss:.6f}")