import pandas as pd
import torch
import torch.nn as nn
import torch.optim as torch_optim
from torch.utils.data import TensorDataset, DataLoader
from attention_model import DataProcessor, TransformerModel, load_model
from plots import plot_loss, plot_sequences
import numpy as np

# Load data
data = pd.read_csv('/home/skillissue/Summer25/World Energy /data/processed/final_model_ready.csv')

# Process data
processor = DataProcessor(data)
processor.filter_data('United States', 'Net Electricity Production', 'Electricity')
processor.sort_data()
processor.split_data(test_size=0.2)
processor.standardize()
train_sequences, train_targets = processor.get_train_sequences()
test_sequences, test_targets = processor.get_test_sequences()

# Convert to torch tensors
train_sequences = torch.tensor(train_sequences, dtype=torch.float32)
train_targets = torch.tensor(train_targets, dtype=torch.float32)
test_sequences = torch.tensor(test_sequences, dtype=torch.float32)
test_targets = torch.tensor(test_targets, dtype=torch.float32)

# Create DataLoaders
batch_size = 32
train_dataset = TensorDataset(train_sequences, train_targets)
test_dataset = TensorDataset(test_sequences, test_targets)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define model
model = TransformerModel(feature_dim=1, d_model=64, nhead=8, num_layers=2)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch_optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(model, train_loader, test_loader, num_epochs=500):
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for sequences, targets in train_loader:
            sequences = sequences.permute(1, 0, 2)  # (seq_len, batch, feature_dim)
            output = model(sequences)
            loss = criterion(output, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for sequences, targets in test_loader:
                sequences = sequences.permute(1, 0, 2)
                output = model(sequences)
                loss = criterion(output, targets)
                val_loss += loss.item()
            val_loss /= len(test_loader)
            val_losses.append(val_loss)

        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    return train_losses, val_losses

# Train the model
train_losses, val_losses = train_model(model, train_loader, test_loader)

# Save the trained model
model_path = '/home/skillissue/Summer25/World Energy /models/saved_models/transformer_model.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Plot losses and save the figure
plot_loss(train_losses, val_losses, filename='loss_plot.png')

# Get predictions with the trained model
model.eval()
with torch.no_grad():
    train_pred = model(train_sequences.permute(1, 0, 2)).numpy()
    test_pred = model(test_sequences.permute(1, 0, 2)).numpy()

# Inverse transform predictions to original scale
train_pred_original = processor.scaler.inverse_transform(train_pred)
test_pred_original = processor.scaler.inverse_transform(test_pred)

# Prepare data for plotting
full_dates = processor.data['date'].values
full_actual = processor.data['value'].values  # Original values before standardization
train_pred_dates = processor.train_data['date'].iloc[processor.seq_len:].values
test_pred_dates = processor.test_data['date'].iloc[processor.seq_len:].values

# Generate future predictions up to December 2030 using the trained model
last_date = pd.to_datetime(processor.data['date'].iloc[-1])
target_date = pd.to_datetime('2030-12-01')
num_future_steps = (target_date.year - last_date.year) * 12 + (target_date.month - last_date.month)
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=num_future_steps, freq='MS')

# Recursive forecasting with the trained model
future_preds = []
current_sequence = test_sequences[-1].clone().detach()  # (seq_len, 1)
for _ in range(num_future_steps):
    with torch.no_grad():
        prediction = model(current_sequence.unsqueeze(1))  # (1,1)
        future_preds.append(prediction.item())
        # Update sequence by shifting and appending the new prediction
        current_sequence = torch.cat([current_sequence[1:], prediction], dim=0)

# Inverse transform future predictions to original scale
future_pred_original = processor.scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

# Plot all sequences including future predictions and save the figure
plot_sequences(full_dates, full_actual, train_pred_dates, train_pred_original, test_pred_dates, test_pred_original, future_dates, future_pred_original, filename='sequences_plot.png')

# Example: Load the saved model for inference
loaded_model = load_model(model_path, feature_dim=1, d_model=64, nhead=8, num_layers=2)
print("Loaded model for inference")

# Example inference with the loaded model
with torch.no_grad():
    sample_pred = loaded_model(test_sequences[-1:].permute(1, 0, 2)).numpy()
    sample_pred_original = processor.scaler.inverse_transform(sample_pred)
    print(f"Sample prediction from loaded model: {sample_pred_original[0][0]:.4f}")