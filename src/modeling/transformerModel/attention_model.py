import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import math

class DataProcessor:
    def __init__(self, data, seq_len=12):
        self.data = data
        self.seq_len = seq_len
        self.scaler = StandardScaler()

    def filter_data(self, country, parameter, product):
        """Filter data for a specific permutation of country, parameter, and product."""
        self.data = self.data[
            (self.data['country_name'] == country) &
            (self.data['parameter'] == parameter) &
            (self.data['product'] == product)
        ]

    def sort_data(self):
        """Sort data by date."""
        self.data = self.data.sort_values('date')

    def split_data(self, test_size=0.2):
        """Split data into training and testing sets."""
        split_idx = int(len(self.data) * (1 - test_size))
        self.train_data = self.data.iloc[:split_idx]
        self.test_data = self.data.iloc[split_idx:]

    def standardize(self):
        """Standardize the value column using training data statistics."""
        self.train_values = self.scaler.fit_transform(self.train_data['value'].values.reshape(-1, 1))
        self.test_values = self.scaler.transform(self.test_data['value'].values.reshape(-1, 1))

    def create_sequences(self, values):
        """Create input sequences and targets for time series forecasting."""
        sequences = []
        targets = []
        for i in range(len(values) - self.seq_len):
            seq = values[i:i + self.seq_len]
            target = values[i + self.seq_len]
            sequences.append(seq)
            targets.append(target)
        return np.array(sequences).reshape(-1, self.seq_len, 1), np.array(targets).reshape(-1, 1)

    def get_train_sequences(self):
        """Get training sequences and targets."""
        return self.create_sequences(self.train_values)

    def get_test_sequences(self):
        """Get testing sequences and targets."""
        return self.create_sequences(self.test_values)

    def preprocess_categorical(self, columns=['hemisphere', 'production_season']):
        """Preprocess categorical columns using one-hot encoding (for future multivariate use)."""
        return pd.get_dummies(self.data[columns])

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerModel(nn.Module):
    def __init__(self, feature_dim=1, d_model=64, nhead=8, num_layers=2, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_projection = nn.Linear(feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, 1)
        self.d_model = d_model
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.input_projection(src)  # (seq_len, batch, feature_dim) -> (seq_len, batch, d_model)
        src = src * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output[-1])  # (batch, d_model) -> (batch, 1)
        return output

def load_model(model_path, feature_dim=1, d_model=64, nhead=8, num_layers=2, dropout=0.1):
    """Load a saved Transformer model from a file."""
    model = TransformerModel(feature_dim=feature_dim, d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model