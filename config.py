from typing import List, Dict
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Trading pairs configuration
TRADING_PAIRS: List[str] = ['BTCUSDT', 'ETHUSDT']
TIMEFRAMES: List[str] = ['1h', '4h', '1d']

# Model Configuration
MODEL_CONFIG = {
    # Transformer parameters
    'n_head': 8,
    'n_layers': 6,
    'd_model': 512,
    'd_ff': 2048,
    'dropout': 0.1,
    
    # LSTM parameters
    'lstm_hidden_size': 256,
    'lstm_layers': 2,
    'lstm_dropout': 0.2,
    
    # Training parameters
    'batch_size': 32,
    'learning_rate': 0.0001,
    'epochs': 100,
    'early_stopping_patience': 10,
}

# Reinforcement Learning Configuration
RL_CONFIG = {
    'gamma': 0.99,  # Discount factor
    'learning_rate': 0.0003,
    'batch_size': 64,
    'buffer_size': 100000,
    'learning_starts': 1000,
    'train_freq': 1,
}

# Trading Configuration
TRADING_CONFIG = {
    'mode': os.getenv('TRADING_MODE', 'spot'),
    'test_mode': os.getenv('TEST_MODE', 'True').lower() == 'true',
    'max_position_size': float(os.getenv('MAX_POSITION_SIZE', 100)),
    'stop_loss_percentage': float(os.getenv('STOP_LOSS_PERCENTAGE', 2.0)),
    'take_profit_percentage': float(os.getenv('TAKE_PROFIT_PERCENTAGE', 4.0)),
}

# Data Collection Configuration
DATA_CONFIG = {
    'historical_data_days': 365,  # Days of historical data to collect
    'orderbook_depth': 10,        # Depth of orderbook data
    'update_interval': 60,        # Data update interval in seconds
}

# Paths Configuration
PATHS = {
    'data_dir': 'data',
    'models_dir': 'models',
    'logs_dir': 'logs',
} 