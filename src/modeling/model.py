# what to do: generate sequences with 3 months rolling averages per year
# implement the training loop in train.py
# implement the eval in evaluate.py 
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import pdb


# Create synthetic data
data = pd.read_csv(r"/home/skillissue/Summer25/World Energy /data/processed/model_ready.csv")
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def create_sequences(data, country, parameter, product, input_years=10, predict_seasons=8):
    """
    Create LSTM-ready sequences with seasonal aggregation and feature engineering
    
    Args:
        data: Main DataFrame
        country: Target country
        parameter: Electricity parameter
        product: Energy product
        input_years: Years of historical data (default 3)
        predict_seasons: Seasons to predict (default 1)
        
    Returns:
        X: Input sequences (num_samples, input_seasons, num_features)
        y: Target values (num_samples, predict_seasons)
        feature_scaler: Fitted scaler for features
        target_scaler: Fitted scaler for target
    """
    # Filter data for specific country, parameter, product
    country_data = data[
        (data['country_name'] == country) &
        (data['parameter'] == parameter) &
        (data['product'] == product)
    ].sort_values('date').copy()
    
    # Get hemisphere for season definition
    hemisphere = country_data['hemisphere'].iloc[0]
    
    # Define seasons based on hemisphere
    seasons = {
        'Northern': {
            'Winter': [12, 1, 2],
            'Spring': [3, 4, 5],
            'Summer': [6, 7, 8],
            'Fall': [9, 10, 11]
        },
        'Southern': {
            'Summer': [12, 1, 2],
            'Fall': [3, 4, 5],
            'Winter': [6, 7, 8],
            'Spring': [9, 10, 11]
        }
    }[hemisphere]
    
    # Create season column
    month_to_season = {}
    for season, months in seasons.items():
        for month in months:
            month_to_season[month] = season
    country_data['season'] = country_data['month'].map(month_to_season)
    
    # Aggregate by season-year
    seasonal_avg = country_data.groupby(['year', 'season']).agg({
        'value': 'mean',
        'major_production_month': 'last',
        'production_season': 'last',
        'peak_ratio': 'last',
        'peak_consistency': 'last',
        'hemisphere': 'first'
    }).reset_index()
    
    # Sort chronologically
    season_order = list(seasons.keys())
    seasonal_avg['season'] = pd.Categorical(seasonal_avg['season'], categories=season_order, ordered=True)
    seasonal_avg = seasonal_avg.sort_values(['year', 'season'])
    
    # Prepare features and target
    features = seasonal_avg[['value', 'major_production_month', 'peak_ratio', 'peak_consistency']]
    target = seasonal_avg['value']
    
    # Feature preprocessing
    num_features = ['value', 'major_production_month', 'peak_ratio', 'peak_consistency']
    cat_features = ['production_season', 'hemisphere']
    
    preprocessor = ColumnTransformer([
        ('num', MinMaxScaler(), num_features),
        ('cat', OneHotEncoder(), cat_features)
    ])
    
    # Fit and transform features
    processed_features = preprocessor.fit_transform(seasonal_avg)
    
    # Create sequences
    input_seasons = input_years * 4  # 4 seasons per year
    total_seasons = len(seasonal_avg)
    X, y = [], []
    
    for i in range(total_seasons - input_seasons - predict_seasons + 1):
        X.append(processed_features[i:i+input_seasons])
        y.append(target.iloc[i+input_seasons:i+input_seasons+predict_seasons].values)
        
    return (np.array(X), 
            np.array(y), 
            preprocessor,
            MinMaxScaler().fit(target.values.reshape(-1, 1)))


import torch
import torch.nn as nn
import torch.optim as optim

class SeasonalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(SeasonalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# Initialize model
def init_model(input_size, output_size):
    return SeasonalLSTM(
        input_size=input_size,
        hidden_size=128,
        output_size=output_size,
        num_layers=2
    )


def train_model(data, country, parameter, product):
    # Create sequences
    X, y, feature_scaler, target_scaler = create_sequences(
        data, country, parameter, product,
        input_years=3,  # 3 years historical data
        predict_seasons=1  # Predict next season
    )
    
    # Train-test split (pre-2020 vs post-2020)
    split_idx = int(len(X) * 0.8)  # 80% train, 20% test
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
    
    # Initialize model
    model = init_model(
        input_size=X_train.shape[2],
        output_size=y_train.shape[1]
    )
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    # Training loop
    num_epochs = 5000
    train_losses, test_losses = [], []
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            
        scheduler.step(test_loss)
        
        # Store losses
        train_losses.append(loss.item())
        test_losses.append(test_loss.item())
        
        # Print progress
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs} | '
                  f'Train Loss: {loss.item():.6f} | '
                  f'Test Loss: {test_loss.item():.6f} |'
                  f'overfitting: {- float(loss.item()) + float(test_loss.item())}' )
            
    
    return model, train_losses, test_losses, feature_scaler, target_scaler

# Example usage
model, train_loss, test_loss, feat_scaler, target_scaler = train_model(
    data, 
    country='Germany',
    parameter='Net Electricity Production',
    product='Electricity'
)

#from ray import tune

# def tune_model(config):
#     model = SeasonalLSTM(
#         input_size=X_train.shape[2],
#         output_size=y_train.shape[1],
#         hidden_size=config["hidden_size"],
#         num_layers=config["num_layers"]
#     )
#     # Training loop with config["lr"]
#     # Return validation loss