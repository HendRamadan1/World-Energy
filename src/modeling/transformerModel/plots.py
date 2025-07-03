import matplotlib.pyplot as plt
import pandas as pd

def plot_loss(train_losses, val_losses, filename='loss_plot.png'):
    """Plot training and validation losses and save the figure."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.savefig(filename)
    plt.show()

def plot_sequences(dates, actual, train_dates, train_pred, test_dates, test_pred, future_dates=None, future_pred=None, filename='sequences_plot.png'):
    """Plot actual, training predicted, testing predicted, and future predicted sequences and save the figure."""
    dates = pd.to_datetime(dates)
    train_dates = pd.to_datetime(train_dates)
    test_dates = pd.to_datetime(test_dates)
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual, label='Actual', color='blue')
    plt.plot(train_dates, train_pred, label='Train Prediction', color='green')
    plt.plot(test_dates, test_pred, label='Test Prediction', color='red')
    if future_dates is not None and future_pred is not None:
        future_dates = pd.to_datetime(future_dates)
        plt.plot(future_dates, future_pred, label='Future Prediction', color='purple')
    plt.xlabel('Date')
    plt.ylabel('Electricity Production (Value)')
    plt.title('Electricity Production: Actual vs Predicted')
    plt.legend()
    plt.savefig(filename)
    plt.show()