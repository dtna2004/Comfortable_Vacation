import torch
import torch.nn as nn
import math
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_CONFIG

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Khởi tạo Positional Encoding
        
        Args:
            d_model: Kích thước của model dimension
            dropout: Tỷ lệ dropout
            max_len: Độ dài tối đa của chuỗi
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, d_model)
            
        Returns:
            Output tensor of shape (batch_size, sequence_length, d_model)
        """
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

class TransformerLSTM(nn.Module):
    def __init__(self, input_size: int, d_model: int, nhead: int, num_layers: int,
                 dim_feedforward: int, dropout: float, lstm_hidden_size: int,
                 lstm_layers: int, lstm_dropout: float):
        """
        Khởi tạo mô hình Transformer-LSTM
        
        Args:
            input_size: Số features đầu vào
            d_model: Kích thước của model dimension trong Transformer
            nhead: Số attention heads
            num_layers: Số Transformer layers
            dim_feedforward: Kích thước của feedforward network trong Transformer
            dropout: Tỷ lệ dropout cho Transformer
            lstm_hidden_size: Kích thước hidden state của LSTM
            lstm_layers: Số LSTM layers
            lstm_dropout: Tỷ lệ dropout cho LSTM
        """
        super().__init__()
        
        # Linear layer để chuyển đổi input features thành d_model dimensions
        self.input_proj = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
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
            num_layers=num_layers
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            dropout=lstm_dropout if lstm_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer
        self.output = nn.Linear(lstm_hidden_size, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # Project input to d_model dimensions
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # LSTM layers
        lstm_out, _ = self.lstm(x)
        
        # Lấy output của time step cuối cùng
        last_hidden = lstm_out[:, -1, :]
        
        # Output layer
        out = self.output(last_hidden)
        
        return out

class TradingRLModel(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        """
        Khởi tạo model Reinforcement Learning cho trading
        
        Args:
            state_dim: Kích thước của state
            action_dim: Số actions có thể thực hiện
        """
        super().__init__()
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, state):
        """
        Forward pass của model
        
        Args:
            state: State tensor
            
        Returns:
            Tuple của (action_probs, state_value)
        """
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value

def create_transformer_lstm_model(input_dim: int) -> TransformerLSTM:
    """
    Tạo instance của TransformerLSTM model
    
    Args:
        input_dim: Số features đầu vào
        
    Returns:
        TransformerLSTM model
    """
    model = TransformerLSTM(input_size=input_dim, d_model=MODEL_CONFIG['d_model'], nhead=MODEL_CONFIG['n_head'], num_layers=MODEL_CONFIG['n_layers'], dim_feedforward=MODEL_CONFIG['d_ff'], dropout=MODEL_CONFIG['dropout'], lstm_hidden_size=MODEL_CONFIG['lstm_hidden_size'], lstm_layers=MODEL_CONFIG['lstm_layers'], lstm_dropout=MODEL_CONFIG['lstm_dropout'])
    return model

def create_rl_model(state_dim: int, action_dim: int) -> TradingRLModel:
    """
    Tạo instance của TradingRLModel
    
    Args:
        state_dim: Kích thước của state
        action_dim: Số actions có thể thực hiện
        
    Returns:
        TradingRLModel
    """
    model = TradingRLModel(state_dim=state_dim, action_dim=action_dim)
    return model 