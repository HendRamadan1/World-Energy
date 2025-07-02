import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from attention_model import ModelFactory, create_sequences
from torch.utils.data import DataLoader, TensorDataset
import os

def preprocess_data(data_path):
    # Load and filter data
    df = pd.read_csv(data_path)
    df = df[(df['country_name'] == 'United States') & 
            (df['parameter'] == 'Net Electricity Production') & 
            (df['product'] == 'Electricity')].sort_values('date')

    # Split into train (2010-2018) and test (2019+)
    train_df = df[df['year'].between(2010, 2018)]
    test_df = df[df['year'] >= 2019]

    # Define features
    categorical_cols = ['country_name', 'parameter', 'product', 'hemisphere', 'production_season']
    numerical_cols = ['year', 'month', 'major_production_month', 'peak_ratio', 'peak_consistency', 
                      'GDP', 'GDP_per_capita', 'Monthly Temperature Averages', 'Daylight Hours', 
                      'Nuclear Plant Status (Inferred Outage)', 'Industrial Production Index', 
                      'Average Energy Price (USD/MWh)']

    # Preprocessing pipeline
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols),
        ('num', StandardScaler(), numerical_cols)
    ])
    target_scaler = StandardScaler()

    # Fit and transform features
    X_train = preprocessor.fit_transform(train_df)
    X_test = preprocessor.transform(test_df)
    y_train = target_scaler.fit_transform(train_df[['value']])
    y_test = target_scaler.transform(test_df[['value']])

    # Combine features and target for sequence creation
    train_data = np.hstack((y_train, X_train))
    test_data = np.hstack((y_test, X_test))
    full_data = np.vstack((train_data, test_data))

    return full_data, train_data.shape[0], preprocessor, target_scaler

def train_model(data_path, save_dir="models"):
    # Preprocess data
    full_data, train_size, preprocessor, target_scaler = preprocess_data(data_path)
    X, y = create_sequences(full_data, seq_length=96, predict_steps=1)

    # Split into train and test
    X_train, X_test = X[:train_size-96], X[train_size-96:]
    y_train, y_test = y[:train_size-96], y[train_size-96:]

    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)

    # DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize model
    input_dim = X_train.shape[2]
    model = ModelFactory.create_model(input_dim=input_dim, output_dim=1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    # Training loop
    train_losses, test_losses = [], []
    for epoch in range(500):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            train_pred = model(X_train)
            test_pred = model(X_test)
            train_loss = criterion(train_pred, y_train).item()
            test_loss = criterion(test_pred, y_test).item()
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")

    # Save model
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{save_dir}/transformer_model.pth")
    return model, train_losses, test_losses, full_data, train_size, target_scaler

def autoregressive_predict(model, data, train_size, horizon, seq_length=96):
    model.eval()
    predictions = []
    input_seq = torch.FloatTensor(data[train_size-seq_length:train_size]).unsqueeze(0)
    
    with torch.no_grad():
        for _ in range(horizon):
            pred = model(input_seq)
            predictions.append(pred.item())
            # Shift sequence and append prediction
            pred_scaled = torch.FloatTensor([[pred.item()] + [0]*(input_seq.shape[2]-1)]).unsqueeze(1)
            input_seq = torch.cat((input_seq[:, 1:, :], pred_scaled), dim=1)
    
    return np.array(predictions)

if __name__ == "__main__":
    data_path = "/home/skillissue/Summer25/World Energy /data/processed/final_model_ready.csv"
    model, train_losses, test_losses, full_data, train_size, target_scaler = train_model(data_path)
    # Predict for the test period and beyond (e.g., until 2030, 132 months from 2019)
    predictions = autoregressive_predict(model, full_data, train_size, horizon=132)