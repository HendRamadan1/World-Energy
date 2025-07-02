import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from model import create_sequences

def plot_sequences(model, data, country, parameter, product, feature_scaler, target_scaler, input_years=3, predict_seasons=4, save_path=None):
    """
    Plot the actual sequence, trained sequence, and predicted sequence at the monthly level.

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
    """
    if predict_seasons != 1:
        print("Plotting is currently supported only for predict_seasons=1.")
        return

    # Get monthly country data
    country_data = data[
        (data['country_name'] == country) &
        (data['parameter'] == parameter) &
        (data['product'] == product)
    ].sort_values('date').copy()
    T = len(country_data)  # Total months

    # Create sequences (monthly granularity)
    X, y, _, _ = create_sequences(data, country, parameter, product, input_years, predict_seasons)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]

    # Convert to tensors and get predictions
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train).numpy()
        test_pred = model(X_test).numpy()

    # Inverse transform predictions
    train_pred = target_scaler.inverse_transform(train_pred).flatten()
    test_pred = target_scaler.inverse_transform(test_pred).flatten()

    # Monthly input sequence length
    input_seasons = input_years * 12  # Matches create_sequences

    # Plot monthly data
    plt.figure(figsize=(12, 6))
    plt.plot(range(T), country_data['value'], label='Actual Sequence', color='blue')

    # Trained sequence (predictions on training data)
    train_pred_plot = np.full(T, np.nan)
    train_start = input_seasons  # First prediction is for month after input sequence
    train_end = train_start + len(train_pred)
    train_pred_plot[train_start:train_end] = train_pred
    plt.plot(range(T), train_pred_plot, label='Trained Sequence', color='green', linestyle='--')

    # Predicted sequence (predictions on test data)
    test_pred_plot = np.full(T, np.nan)
    test_start = train_end  # Starts right after training predictions
    test_end = test_start + len(test_pred)
    test_pred_plot[test_start:test_end] = test_pred
    plt.plot(range(T), test_pred_plot, label='Predicted Sequence', color='red', linestyle='--')

    plt.xlabel('Month Index')
    plt.ylabel('Value')
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
    plt.title('Train vs Test Loss over Epochs')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()