import torch
import torch.optim as optim
import pandas as pd
import os
from model import SeasonalLSTM, create_sequences
import torch.nn as nn
import pickle
from plots import plot_sequences, plot_loss

class WeightedMSELoss(nn.Module):
    def __init__(self, weight_factor=2.0):
        super(WeightedMSELoss, self).__init__()
        self.weight_factor = weight_factor

    def forward(self, y_pred, y_true):
        error = y_true - y_pred
        weights = torch.abs(error) * self.weight_factor
        return torch.mean(weights * error ** 2)

def train_model(data, country, parameter, product, hidden_size=128, num_layers=2, lr=0.001, weight_decay=1e-5, num_epochs=7000):
    """Train the SeasonalLSTM model.

    Args:
        data (pd.DataFrame): Input dataset.
        country (str): Country to filter data.
        parameter (str): Parameter to filter data.
        product (str): Product to filter data.
        hidden_size (int): Number of hidden units. Defaults to 128.
        num_layers (int): Number of LSTM layers. Defaults to 2.
        lr (float): Learning rate. Defaults to 0.001.
        weight_decay (float): Weight decay for regularization. Defaults to 1e-5.
        num_epochs (int): Number of training epochs. Defaults to 5000.

    Returns:
        tuple: (model, train_losses, test_losses, feature_scaler, target_scaler)
    """
    X, y, feature_scaler, target_scaler = create_sequences(data, country, parameter, product)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)

    model = SeasonalLSTM(input_size=X_train.shape[2], hidden_size=hidden_size, output_size=y_train.shape[1], num_layers=num_layers, dropout=0.2)

    criterion = WeightedMSELoss(weight_factor=2.0)
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

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {loss.item():.6f} | Test Loss: {test_loss.item():.6f} | Overfitting: {test_loss.item() - loss.item():.6f}')

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
        data,
        country='Argentina',
        parameter='Net Electricity Production',
        product='Electricity',
        hidden_size=256,
        num_layers=1,
        lr=0.0007221676729397737,
        weight_decay=0.000230930982691119,
        num_epochs=1000
    )
    # Best Hyperparameters: {'hidden_size': 128, 'num_layers': 2, 'learning_rate': 0.0030667674041639148, 'weight_decay': 0.00021075022059945686, 'dropout': 0.3}

    # Plot sequences and losses
    plot_sequences(
        model=model,
        data=data,
        country='Argentina',
        parameter='Net Electricity Production',
        product='Electricity',
        feature_scaler=feat_scaler,
        target_scaler=target_scaler,
        input_years=3,
        predict_seasons=4,
        save_path="/home/skillissue/Summer25/World Energy /plots/sequences.png"
    )
    plot_loss(
        train_losses=train_losses,
        test_losses=test_losses,
        save_path="/home/skillissue/Summer25/World Energy /plots/losses.png"
    )