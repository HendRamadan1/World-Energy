import streamlit as st

st.markdown("# Attention Mechanism (Transformer)")
st.sidebar.markdown("""
                    _Transformers are neural networks that process sequential data using self-attention, allowing them to weigh input elements dynamically. Unlike RNNs, they handle long-range dependencies efficiently in parallel. \n learn more about transformers: [click here](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture))_""")
st.markdown("""in this project, our main goal was to predict electricity production values given all the past values, in the data, the values are segmented by month, from `1/2010` to `12/2023` across 48 countries, we picked the USA to predict its electricity production, given its impact on global economy and European Energy Crisis after the halt on russian energy imports.

### Our goal
- predict energy production till the year 2030
- include feature engineered columns in the data
- make the model efficient and quick

### Code Implementation

####  Overview
This implements a Transformer model with positional encoding for sequence processing. The model consists of two main components:
1. `PositionalEncoding` - Adds positional information to embeddings
2. `TransformerModel` - The complete transformer architecture

#### Positional Encoding

```python
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
```

#### Key Features:
- **Sinusoidal Patterns**: Uses alternating sine and cosine functions to encode position information
- **Frequency Calculation**: 
  - `div_term` creates geometrically decreasing frequencies
  - Higher dimensions get smaller wavelength patterns
- **Memory Efficient**: Pre-computes encodings for all positions up to `max_len`

#### Why This Works:
- Allows model to learn relative positions through wave patterns
- Fixed (non-learned) encoding works well in practice
- Handles sequences longer than those seen during training

#### Transformer Model

```python
class TransformerModel(nn.Module):
    def __init__(self, feature_dim=1, d_model=64, nhead=8, num_layers=2, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_projection = nn.Linear(feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, 1)
```

#### Architecture Components:
1. **Input Projection**:
   - Linear layer to map input features to model dimension
   - Scales by √d_model to preserve magnitude

2. **Positional Encoding**:
   - Adds crucial sequence order information

3. **Transformer Encoder**:
   - Stack of `num_layers` identical layers
   - Each layer has:
     - Multi-head self-attention
     - Position-wise feedforward network
     - Residual connections & layer normalization

4. **Decoder Head**:
   - Final linear layer producing output
   - Only uses last timestep's representation

#### Forward Pass Flow:
```mermaid
graph TD
    A[Input] --> B[Project to d_model]
    B --> C[Scale by √d_model]
    C --> D[Add Positional Encoding]
    D --> E[Transformer Encoder Layers]
    E --> F[Last Timestep Only]
    F --> G[Decode to Output]
```

#### Software engineering and Design pattern choices:
- **Feature Dimension Flexibility**: Can handle any input feature size
- **Modular Architecture**: Easy to adjust depth (`num_layers`) and width (`d_model`)
- **Sequence Processing**: Only uses last output (good for forecasting)
- **Initialization**: Careful weight initialization for stable training

""")