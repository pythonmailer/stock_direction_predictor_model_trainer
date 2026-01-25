import torch
import torch.nn as nn
from src.utils import get_logger

class LSTM(nn.Module):
    def __init__(self, d_in, hidden_size, n_layers, dropout=0.2):
        super().__init__()
        self.logger = get_logger(__name__)
        self.logger.info(f"Initializing LSTM with d_in={d_in}, hidden_size={hidden_size}, n_layers={n_layers}, dropout={dropout}")
        
        self.lstm = nn.LSTM(
            input_size=d_in, 
            hidden_size=hidden_size, 
            num_layers=n_layers, 
            batch_first=True, 
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_step_out = out[:, -1, :]
        x = self.dropout(last_step_out)
        return self.fc(x)