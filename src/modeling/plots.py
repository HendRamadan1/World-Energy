# import matplotlib.pyplot as plt
import torch
import numpy as np
from model import create_sequences  # Assuming create_sequences is defined in model.py
import matplotlib.pyplot as plt
import pandas as pd

def plot_sequences(model, data, country, parameter, product, feature_scaler, target_scaler, input_years=3, predict_seasons=1, save_path=None):
    """
    Plot the actual sequence, trained sequence, and predicted sequence.

    Args:
        model (SeasonalLSTM): Trained model.
        data (pd.DataFrame): The complete dataset.
        country (str): Country to filter data.
        parameter (str): Parameter to filter data.
        product (str): Product to filter data.
        feature_scaler (ColumnTransformer): Scaler for features.
        target_scaler (MinMaxScaler): Scaler for target values.
        input_years (int): Number of years for input sequence. Defaults to 3.
        predict_seasons (int): Number of seasons to predict. Defaults to 1.
        save_path (str, optional): Path to save the plot. If None, the plot is displayed.

    Note:
        This function assumes predict_seasons=1 for simplicity. For predict_seasons > 1, 
        the plotting logic may need to be adjusted.
    """
    if predict_seasons != 1:
        print("Plotting is currently supported only for predict_seasons=1.")
        return

    # Create sequences
    X, y, _, _ = create_sequences(data, country, parameter, product, input_years, predict_seasons)
    
    # Get seasonal_avg from create_sequences logic
    country_data = data[
        (data['country_name'] == country) &
        (data['parameter'] == parameter) &
        (data['product'] == product)
    ].sort_values('date').copy()
    country_data['rolling_avg'] = country_data['value'].rolling(window=3, min_periods=1).mean()
    country_data = country_data.dropna()
    hemisphere = country_data['hemisphere'].iloc[0]
    seasons = {
        'Northern': {'Winter': [12, 1, 2], 'Spring': [3, 4, 5], 'Summer': [6, 7, 8], 'Fall': [9, 10, 11]},
        'Southern': {'Summer': [12, 1, 2], 'Fall': [3, 4, 5], 'Winter': [6, 7, 8], 'Spring': [9, 10, 11]}
    }[hemisphere]
    month_to_season = {month: season for season, months in seasons.items() for month in months}
    country_data['season'] = country_data['month'].map(month_to_season)
    seasonal_avg = country_data.groupby(['year', 'season']).agg({'rolling_avg': 'mean'}).reset_index()
    season_order = list(seasons.keys())
    seasonal_avg['season'] = pd.Categorical(seasonal_avg['season'], categories=season_order, ordered=True)
    seasonal_avg = seasonal_avg.sort_values(['year', 'season'])

    # Split into train and test
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train).numpy()
        test_pred = model(X_test).numpy()
    
    # Inverse transform predictions
    train_pred = target_scaler.inverse_transform(train_pred).flatten()
    test_pred = target_scaler.inverse_transform(test_pred).flatten()
    
    # Calculate input_seasons
    input_seasons = input_years * 4  # Assuming 4 seasons per year
    
    # Plot
    plt.figure(figsize=(12, 6))
    total_seasons = len(seasonal_avg)
    plt.plot(range(total_seasons), seasonal_avg['rolling_avg'], label='Actual Sequence', color='blue')
    
    # Plot trained sequence (predictions on training data)
    train_start = input_seasons
    train_end = train_start + len(train_pred)
    plt.plot(range(train_start, train_end), train_pred, label='Trained Sequence', color='green', linestyle='--')
    
    # Plot predicted sequence (predictions on test data)
    test_start = input_seasons + split_idx
    test_end = test_start + len(test_pred)
    plt.plot(range(test_start, test_end), test_pred, label='Predicted Sequence', color='red', linestyle='--')
    
    plt.xlabel('Season Index')
    plt.ylabel('Rolling Average')
    plt.title('Sequence Comparison')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_loss(train_losses, test_losses, save_path=None):
    """
    Plot the train loss and test loss over epochs.

    Args:
        train_losses (list): List of train losses over epochs.
        test_losses (list): List of test losses over epochs.
        save_path (str, optional): Path to save the plot. If None, the plot is displayed.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(test_losses, label='Test Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs Test Loss over 5000 Epochs')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()