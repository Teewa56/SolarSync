import torch
import torch.nn as nn

class LSTMForecastModel(nn.Module):
    """
    LSTM-based model for time series prediction (e.g., solar energy generation).
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        """
        Args:
            input_size: Number of features in the input sequence.
            hidden_size: Number of features in the hidden state. (Specified as 128)
            num_layers: Number of recurrent layers. (Specified as 3)
            output_size: Number of values to predict (1 for the next timestep).
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Define the LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True  # Input tensor will be (batch_size, sequence_length, input_size)
        )
        
        # Define the output layer (maps LSTM output to the prediction)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the model.
        
        Args:
            x: Input tensor (batch_size, sequence_length, input_size).
            
        Returns:
            Output tensor (batch_size, output_size).
        """
        # Initialize hidden and cell states (can be omitted if PyTorch initializes them)
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Pass input through LSTM
        # out will have shape (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(x) #, (h0, c0))
        
        # Take the output from the last time step
        # out[:, -1, :] has shape (batch_size, hidden_size)
        out = self.fc(out[:, -1, :])
        
        return out