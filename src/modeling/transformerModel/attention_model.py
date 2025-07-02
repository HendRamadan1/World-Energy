import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2, n_heads=4, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        x = self.embedding(x)  # (batch_size, seq_len, hidden_dim)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # (batch_size, seq_len, hidden_dim)
        x = self.fc(x[:, -1, :])  # Predict from last timestep
        return x

class ModelFactory:
    @staticmethod
    def create_model(input_dim, output_dim, hidden_dim=128, n_layers=2, n_heads=4, dropout=0.1):
        return TransformerModel(input_dim, hidden_dim, output_dim, n_layers, n_heads, dropout)

def create_sequences(data, seq_length=96, predict_steps=1):
    X, y = [], []
    for i in range(len(data) - seq_length - predict_steps + 1):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length:i + seq_length + predict_steps, 0])  # 'value' is first column
    return np.array(X), np.array(y)