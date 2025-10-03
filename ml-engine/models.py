import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMForecastModel(nn.Module):
    """
    LSTM-based model for time series prediction (solar energy generation)
    """
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        num_layers: int, 
        output_size: int,
        dropout: float = 0.2
    ):
        """
        Args:
            input_size: Number of features in input sequence
            hidden_size: Number of features in hidden state (128)
            num_layers: Number of recurrent layers (3)
            output_size: Number of values to predict (1 for next timestep)
            dropout: Dropout probability for regularization
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer with dropout
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Batch normalization for stability
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor (batch_size, output_size)
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Take the output from the last time step
        last_output = lstm_out[:, -1, :]
        
        # Apply batch normalization (only if batch size > 1)
        if last_output.size(0) > 1:
            last_output = self.batch_norm(last_output)
        
        # Apply dropout
        last_output = self.dropout(last_output)
        
        # Fully connected layers with ReLU activation
        out = F.relu(self.fc1(last_output))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class GRUForecastModel(nn.Module):
    """
    GRU-based model with attention mechanism for wind energy prediction
    """
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        num_layers: int, 
        output_size: int,
        dropout: float = 0.2
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = AttentionLayer(hidden_size)
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention
        
        Args:
            x: Input tensor (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor (batch_size, output_size)
        """
        # GRU forward pass
        gru_out, hidden = self.gru(x)
        
        # Apply attention to all timesteps
        attention_out = self.attention(gru_out)
        
        # Batch normalization (only if batch size > 1)
        if attention_out.size(0) > 1:
            attention_out = self.batch_norm(attention_out)
        
        # Dropout
        attention_out = self.dropout(attention_out)
        
        # Fully connected layers
        out = F.relu(self.fc1(attention_out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class AttentionLayer(nn.Module):
    """
    Attention mechanism for sequence modeling
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention_weights = nn.Linear(hidden_size, 1)
        
    def forward(self, gru_output: torch.Tensor) -> torch.Tensor:
        """
        Apply attention to GRU outputs
        
        Args:
            gru_output: Tensor (batch_size, sequence_length, hidden_size)
            
        Returns:
            Weighted output (batch_size, hidden_size)
        """
        # Calculate attention scores
        attention_scores = self.attention_weights(gru_output)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Apply attention weights to GRU outputs
        weighted_output = torch.sum(gru_output * attention_weights, dim=1)
        
        return weighted_output


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input
        """
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)


class TransformerForecastModel(nn.Module):
    """
    Transformer-based model for advanced time series forecasting
    """
    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        output_size: int = 1
    ):
        super().__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Output layers
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        # Project input to d_model dimensions
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Transformer encoding
        transformer_out = self.transformer_encoder(x)
        
        # Take the last timestep output
        last_output = transformer_out[:, -1, :]
        
        # Output layers
        out = F.relu(self.fc1(last_output))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out