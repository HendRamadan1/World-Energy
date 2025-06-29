import torch
import torch.nn as nn
import pandas as pd
import pickle
import os
from model import SeasonalLSTM, create_sequences

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on the test set.

    Args:
        model (SeasonalLSTM): Trained model.
        X_test (torch.Tensor): Test input data.
        y_test (torch.Tensor): Test target data.

    Returns:
        float: Test loss.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        criterion = nn.MSELoss()
        loss = criterion(outputs, y_test)
    return loss.item()

if __name__ == "__main__":
    data = pd.read_csv(r"/home/skillissue/Summer25/World Energy /data/processed/model_ready.csv")
    X, y, feature_scaler, target_scaler = create_sequences(
        data, 'Germany', 'Net Electricity Production', 'Electricity'
    )

    split_idx = int(len(X) * 0.8)
    X_test = torch.FloatTensor(X[split_idx:])
    y_test = torch.FloatTensor(y[split_idx:])

    params_path = "/home/skillissue/Summer25/World Energy /models/saved_models/best_params.pkl"
    with open(params_path, "rb") as f:
        best_params = pickle.load(f)

    model = SeasonalLSTM(
        input_size=X_test.shape[2],
        hidden_size=best_params['hidden_size'],
        output_size=y_test.shape[1],
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout']
    )

    model_path = "/home/skillissue/Summer25/World Energy /models/saved_models/tuned_seasonal_lstm_model.pth"
    model.load_state_dict(torch.load(model_path))

    test_loss = evaluate_model(model, X_test, y_test)
    print(f"Tuned Model Test Loss: {test_loss:.6f}")