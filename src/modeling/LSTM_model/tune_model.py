import torch
import torch.optim as optim
import pandas as pd
import optuna
from torch.utils.data import TensorDataset, DataLoader
from model import SeasonalLSTM, create_sequences
import pickle
import os
import torch.nn as nn

def objective(trial, X_train, y_train, X_val, y_val, train_dataloader):
    """Objective function for hyperparameter tuning with Optuna.

    Args:
        trial (optuna.Trial): Optuna trial object.
        X_train (torch.Tensor): Training input data.
        y_train (torch.Tensor): Training target data.
        X_val (torch.Tensor): Validation input data.
        y_val (torch.Tensor): Validation target data.
        train_dataloader (DataLoader): DataLoader for training data.

    Returns:
        float: Validation loss.
    """
    hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
    num_layers = trial.suggest_categorical('num_layers', [1, 2, 3])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)
    dropout = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3])

    model = SeasonalLSTM(
        input_size=X_train.shape[2],
        hidden_size=hidden_size,
        output_size=y_train.shape[1],
        num_layers=num_layers,
        dropout=dropout
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience = 10
    counter = 0

    for epoch in range(1000):
        model.train()
        for batch_X, batch_y in train_dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    return best_val_loss

def tune_and_train(data, country, parameter, product):
    """Tune hyperparameters and train the final model.

    Args:
        data (pd.DataFrame): Input dataset.
        country (str): Country to filter data.
        parameter (str): Parameter to filter data.
        product (str): Product to filter data.

    Returns:
        tuple: (model, best_params, feature_scaler, target_scaler)
    """
    X, y, feature_scaler, target_scaler = create_sequences(data, country, parameter, product)

    n = len(X)
    train_idx = int(n * 0.6)
    val_idx = int(n * 0.8)

    X_train = torch.FloatTensor(X[:train_idx])
    y_train = torch.FloatTensor(y[:train_idx])
    X_val = torch.FloatTensor(X[train_idx:val_idx])
    y_val = torch.FloatTensor(y[train_idx:val_idx])
    X_test = torch.FloatTensor(X[val_idx:])
    y_test = torch.FloatTensor(y[val_idx:])

    train_dataset = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, train_dataloader), n_trials=50)

    best_params = study.best_params
    print("Best Hyperparameters:", best_params)

    X_train_final = torch.cat((X_train, X_val), dim=0)
    y_train_final = torch.cat((y_train, y_val), dim=0)
    train_final_dataset = TensorDataset(X_train_final, y_train_final)
    train_final_dataloader = DataLoader(train_final_dataset, batch_size=16, shuffle=True)

    model = SeasonalLSTM(
        input_size=X_train.shape[2],
        hidden_size=best_params['hidden_size'],
        output_size=y_train.shape[1],
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout']
    )
    optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    criterion = nn.MSELoss()

    best_test_loss = float('inf')
    patience = 10
    counter = 0

    for epoch in range(5000):
        model.train()
        for batch_X, batch_y in train_final_dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test).item()

        scheduler.step(test_loss)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            counter = 0
            torch.save(model.state_dict(), "/home/skillissue/Summer25/World Energy/models/saved_models/tuned_seasonal_lstm_model.pth")
        else:
            counter += 1
            if counter >= patience:
                break

    print(f"Final Test Loss: {best_test_loss:.6f}")

    save_dir = "/home/skillissue/Summer25/World Energy/models/saved_models"
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "best_params.pkl"), "wb") as f:
        pickle.dump(best_params, f)
    with open(os.path.join(save_dir, "tuned_feature_scaler.pkl"), "wb") as f:
        pickle.dump(feature_scaler, f)
    with open(os.path.join(save_dir, "tuned_target_scaler.pkl"), "wb") as f:
        pickle.dump(target_scaler, f)

    return model, best_params, feature_scaler, target_scaler

if __name__ == "__main__":
    data = pd.read_csv(r"/home/skillissue/Summer25/World Energy /data/processed/model_ready.csv")
    model, best_params, feature_scaler, target_scaler = tune_and_train(
        data, 'United States', 'Net Electricity Production', 'Electricity'
    )