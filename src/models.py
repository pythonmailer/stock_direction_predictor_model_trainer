import torch
import torch.nn as nn
import numpy as np
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

class TransformerEncoder(nn.Module):
    def __init__(self, d_in, d_model, nhead, n_layers, d_out, max_len=100, dropout=0.2, ratio_val=None):
        super().__init__()
        self.logger = get_logger(__name__)
        self.logger.info(f"Initializing TransformerEncoder with d_in={d_in}, d_model={d_model}, nhead={nhead}, n_layers={n_layers}, d_out={d_out}, max_len={max_len}, dropout={dropout}, pos_weight_val={pos_weight_val}")
        if d_model % nhead != 0:
            msg = "d_model must be divisible by nhead"
            self.logger.error(msg)
            raise ValueError(msg)

        self.project = nn.Linear(d_in, d_model)

        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        self.emb_dropout = nn.Dropout(dropout)
        self.final_dropout = nn.Dropout(dropout)
        
        self.out = nn.Linear(d_model, d_out)
        
        if ratio_val is not None:
            init_bias = -np.log(ratio_val)
            self.logger.info(f"Initial Bias -> {init_bias}")
            with torch.no_grad():
                self.out.bias.fill_(init_bias)

    def forward(self, x):
        x = self.project(x) + self.pos_emb[:, :x.size(1)]
        x = self.emb_dropout(x)
        x = self.encoder(x)
        x = x[:, -1, :]
        x = self.final_dropout(x)
        return self.out(x)