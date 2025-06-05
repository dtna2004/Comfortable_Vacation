import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRADING_PAIRS, TIMEFRAMES, MODEL_CONFIG, PATHS
from model import TransformerLSTM

# Thiết lập logging
os.makedirs(PATHS['logs_dir'], exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PATHS['logs_dir'], 'training.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CryptoDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ModelTrainer:
    def __init__(self, model_config=MODEL_CONFIG):
        """Khởi tạo ModelTrainer"""
        self.config = model_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Tạo thư mục models nếu chưa tồn tại
        os.makedirs(PATHS['models_dir'], exist_ok=True)
        
    def load_data(self, symbol: str, interval: str) -> tuple:
        """
        Load dữ liệu đã xử lý
        
        Args:
            symbol: Cặp giao dịch
            interval: Khung thời gian
            
        Returns:
            Tuple của (X_train, y_train, X_val, y_val)
        """
        try:
            # Load dữ liệu
            X = np.load(os.path.join(PATHS['data_dir'], f"{symbol}_{interval}_X.npy"))
            y = np.load(os.path.join(PATHS['data_dir'], f"{symbol}_{interval}_y.npy"))
            
            # Chia tập train/val
            train_size = int(0.8 * len(X))
            X_train, X_val = X[:train_size], X[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]
            
            logger.info(f"Loaded data for {symbol} {interval}")
            logger.info(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
            logger.info(f"Val shapes: X={X_val.shape}, y={y_val.shape}")
            
            return X_train, y_train, X_val, y_val
            
        except Exception as e:
            logger.error(f"Error loading data for {symbol} {interval}: {str(e)}")
            return None, None, None, None
            
    def train_model(self, symbol: str, interval: str):
        """
        Huấn luyện mô hình cho một cặp giao dịch và timeframe
        
        Args:
            symbol: Cặp giao dịch
            interval: Khung thời gian
        """
        try:
            # Load dữ liệu
            X_train, y_train, X_val, y_val = self.load_data(symbol, interval)
            if X_train is None:
                return
                
            # Tạo dataloaders
            train_dataset = CryptoDataset(X_train, y_train)
            val_dataset = CryptoDataset(X_val, y_val)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['batch_size'],
                shuffle=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['batch_size']
            )
            
            # Khởi tạo model
            input_size = X_train.shape[2]  # Số features
            model = TransformerLSTM(
                input_size=input_size,
                d_model=self.config['d_model'],
                nhead=self.config['n_head'],
                num_layers=self.config['n_layers'],
                dim_feedforward=self.config['d_ff'],
                dropout=self.config['dropout'],
                lstm_hidden_size=self.config['lstm_hidden_size'],
                lstm_layers=self.config['lstm_layers'],
                lstm_dropout=self.config['lstm_dropout']
            ).to(self.device)
            
            # Loss và optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
            
            # Early stopping
            best_val_loss = float('inf')
            patience = self.config['early_stopping_patience']
            patience_counter = 0
            
            # Training loop
            for epoch in range(self.config['epochs']):
                # Training
                model.train()
                train_loss = 0
                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                train_loss /= len(train_loader)
                
                # Validation
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        outputs = model(batch_X)
                        loss = criterion(outputs.squeeze(), batch_y)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                
                logger.info(f"Epoch {epoch+1}/{self.config['epochs']}")
                logger.info(f"Train Loss: {train_loss:.6f}")
                logger.info(f"Val Loss: {val_loss:.6f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Lưu model tốt nhất
                    model_path = os.path.join(
                        PATHS['models_dir'],
                        f"{symbol}_{interval}_model.pth"
                    )
                    torch.save(model.state_dict(), model_path)
                    logger.info(f"Saved best model to {model_path}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info("Early stopping triggered")
                        break
            
            logger.info(f"Training completed for {symbol} {interval}")
            
        except Exception as e:
            logger.error(f"Error training model for {symbol} {interval}: {str(e)}")

def main():
    """Hàm main để chạy training"""
    try:
        trainer = ModelTrainer()
        for symbol in TRADING_PAIRS:
            for interval in TIMEFRAMES:
                logger.info(f"Training model for {symbol} {interval}")
                trainer.train_model(symbol, interval)
                
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Error in training: {str(e)}")

if __name__ == "__main__":
    main() 