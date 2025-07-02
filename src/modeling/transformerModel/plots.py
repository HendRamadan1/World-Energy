import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd

def plot_loss(train_losses, test_losses, save_path="loss_plot.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(test_losses, label='Test Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Train vs Test Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_sequences(data, train_size, predictions, target_scaler, save_path="sequence_plot.png"):
    actual = target_scaler.inverse_transform(data[:, 0].reshape(-1, 1)).flatten()
    pred_scaled = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    
    plt.figure(figsize=(15, 6))
    plt.plot(range(len(actual)), actual, label='Actual', color='blue')
    plt.plot(range(train_size, train_size + len(pred_scaled)), pred_scaled, label='Predicted', color='red', linestyle='--')
    plt.axvline(x=train_size, color='gray', linestyle='--', label='Train/Test Split')
    plt.xlabel('Month Index')
    plt.ylabel('Electricity Production')
    plt.title('Actual vs Predicted Sequence (2019-2030)')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    from train import train_model, autoregressive_predict
    data_path = "/home/skillissue/Summer25/World Energy /data/processed/final_model_ready.csv"
    model, train_losses, test_losses, full_data, train_size, target_scaler = train_model(data_path)
    predictions = autoregressive_predict(model, full_data, train_size, horizon=132)
    plot_loss(train_losses, test_losses)
    plot_sequences(full_data, train_size, predictions, target_scaler)