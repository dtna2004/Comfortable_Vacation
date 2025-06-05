import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from typing import Dict, List, Tuple
import json
from sklearn.model_selection import ParameterGrid
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRADING_PAIRS, TIMEFRAMES, PATHS, MODEL_CONFIG
from model import TransformerLSTM
from utils import prepare_live_data, load_model
from training import CryptoDataset

# Thiết lập logging
os.makedirs(PATHS['logs_dir'], exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PATHS['logs_dir'], 'optimization.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelOptimizer:
    def __init__(self):
        """Khởi tạo Model Optimizer"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Tạo thư mục cho optimization results
        self.optim_dir = os.path.join(PATHS['data_dir'], 'optimization_results')
        os.makedirs(self.optim_dir, exist_ok=True)
        
    def load_data(self, symbol: str, interval: str) -> Tuple[DataLoader, DataLoader]:
        """
        Load dữ liệu training và validation
        
        Args:
            symbol: Trading pair
            interval: Timeframe
            
        Returns:
            Tuple của (train_loader, val_loader)
        """
        try:
            # Load dữ liệu
            X = np.load(os.path.join(PATHS['data_dir'], f"{symbol}_{interval}_X.npy"))
            y = np.load(os.path.join(PATHS['data_dir'], f"{symbol}_{interval}_y.npy"))
            
            # Chia tập train/val
            train_size = int(0.8 * len(X))
            X_train, X_val = X[:train_size], X[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]
            
            # Tạo dataloaders
            train_dataset = CryptoDataset(X_train, y_train)
            val_dataset = CryptoDataset(X_val, y_val)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=MODEL_CONFIG['batch_size'],
                shuffle=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=MODEL_CONFIG['batch_size']
            )
            
            return train_loader, val_loader
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return None, None
            
    def train_model(self, model: nn.Module, train_loader: DataLoader,
                   val_loader: DataLoader, optimizer: optim.Optimizer,
                   criterion: nn.Module, epochs: int) -> Tuple[float, List[float]]:
        """
        Train model với các hyperparameters cho trước
        
        Args:
            model: Model instance
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer instance
            criterion: Loss function
            epochs: Số epochs
            
        Returns:
            Tuple của (best_val_loss, train_losses)
        """
        model = model.to(self.device)
        best_val_loss = float('inf')
        train_losses = []
        
        for epoch in range(epochs):
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
            train_losses.append(train_loss)
            
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
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
            logger.info(f"Epoch {epoch+1}/{epochs}")
            logger.info(f"Train Loss: {train_loss:.6f}")
            logger.info(f"Val Loss: {val_loss:.6f}")
            
        return best_val_loss, train_losses
        
    def optimize_hyperparameters(self, symbol: str, interval: str) -> Dict:
        """
        Tối ưu hóa hyperparameters cho model
        
        Args:
            symbol: Trading pair
            interval: Timeframe
            
        Returns:
            Dictionary chứa hyperparameters tối ưu
        """
        # Load data
        train_loader, val_loader = self.load_data(symbol, interval)
        if train_loader is None or val_loader is None:
            return {}
            
        # Grid search parameters
        param_grid = {
            'd_model': [64, 128, 256],
            'n_head': [4, 8],
            'n_layers': [2, 3, 4],
            'd_ff': [256, 512],
            'dropout': [0.1, 0.2],
            'lstm_hidden_size': [64, 128],
            'lstm_layers': [1, 2],
            'learning_rate': [0.001, 0.0001]
        }
        
        best_val_loss = float('inf')
        best_params = None
        results = []
        
        # Grid search
        for params in ParameterGrid(param_grid):
            logger.info(f"Testing parameters: {params}")
            
            # Khởi tạo model với params hiện tại
            input_size = next(iter(train_loader))[0].shape[2]  # Số features
            model = TransformerLSTM(
                input_size=input_size,
                d_model=params['d_model'],
                nhead=params['n_head'],
                num_layers=params['n_layers'],
                dim_feedforward=params['d_ff'],
                dropout=params['dropout'],
                lstm_hidden_size=params['lstm_hidden_size'],
                lstm_layers=params['lstm_layers'],
                lstm_dropout=params['dropout']
            )
            
            # Optimizer và loss function
            optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
            criterion = nn.MSELoss()
            
            # Train model
            val_loss, train_losses = self.train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                epochs=MODEL_CONFIG['epochs']
            )
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = params
                
                # Lưu model tốt nhất
                model_path = os.path.join(
                    PATHS['models_dir'],
                    f"{symbol}_{interval}_optimized_model.pth"
                )
                torch.save(model.state_dict(), model_path)
                
            results.append({
                'params': params,
                'val_loss': val_loss,
                'train_losses': train_losses
            })
            
        # Lưu kết quả optimization
        self.save_optimization_results(symbol, interval, best_params, results)
        
        return best_params
        
    def save_optimization_results(self, symbol: str, interval: str,
                                best_params: Dict, results: List[Dict]):
        """Lưu kết quả optimization"""
        results_file = os.path.join(
            self.optim_dir,
            f"model_optimization_{symbol}_{interval}.json"
        )
        
        with open(results_file, 'w') as f:
            json.dump({
                'symbol': symbol,
                'interval': interval,
                'best_params': best_params,
                'all_results': results
            }, f, indent=4)
            
def main():
    """Hàm main để chạy optimization"""
    try:
        optimizer = ModelOptimizer()
        
        # Optimize cho mỗi cặp giao dịch
        for symbol in TRADING_PAIRS:
            for interval in TIMEFRAMES:
                logger.info(f"Optimizing model for {symbol} {interval}")
                
                best_params = optimizer.optimize_hyperparameters(symbol, interval)
                
                if best_params:
                    logger.info(f"Best parameters for {symbol} {interval}:")
                    logger.info(json.dumps(best_params, indent=2))
                    
        logger.info("Optimization completed successfully")
        
    except Exception as e:
        logger.error(f"Error in optimization: {str(e)}")

if __name__ == "__main__":
    main() 