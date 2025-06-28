import torch
import torch.nn as nn
import torch.optim as optim
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

def train_model(data, country, parameter, product, hidden_size=128, num_layers=2, lr=0.001, weight_decay=1e-5, num_epochs=5000):
    X, y, feature_scaler, target_scaler = create_sequences(data, country, parameter, product)
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
    
    model = SeasonalLSTM(input_size=X_train.shape[2], hidden_size=hidden_size, output_size=y_train.shape[1], num_layers=num_layers)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    train_losses, test_losses = [], []
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
        
        scheduler.step(test_loss)
        train_losses.append(loss.item())
        test_losses.append(test_loss.item())
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {loss.item():.6f} | Test Loss: {test_loss.item():.6f} | Overfitting: {test_loss.item() - loss.item():.6f}')
    
    # Save the model and scalers
    save_dir = "/home/skillissue/Summer25/World Energy /models/saved_models"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "seasonal_lstm_model.pth"))
    with open(os.path.join(save_dir, "feature_scaler.pkl"), "wb") as f:
        pickle.dump(feature_scaler, f)
    with open(os.path.join(save_dir, "target_scaler.pkl"), "wb") as f:
        pickle.dump(target_scaler, f)
    
    return model, train_losses, test_losses, feature_scaler, target_scaler

if __name__ == "__main__":
    data = pd.read_csv(r"/home/skillissue/Summer25/World Energy /data/processed/model_ready.csv")
    model, train_losses, test_losses, feat_scaler, target_scaler = train_model(
        data, 'Germany', 'Net Electricity Production', 'Electricity'
    )