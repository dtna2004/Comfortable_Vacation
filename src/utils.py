import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import logging
import sys
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PATHS

# Thiết lập logging
os.makedirs(PATHS['logs_dir'], exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PATHS['logs_dir'], 'utils.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def plot_training_history(history: Dict[str, List[float]], save_path: str) -> None:
    """
    Vẽ đồ thị training history
    
    Args:
        history: Dictionary chứa loss values
        save_path: Đường dẫn để lưu plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Tính toán các metrics đánh giá model
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Dictionary chứa các metrics
    """
    try:
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }
        
        logger.info(f"Calculated metrics: {metrics}")
        return metrics
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        return None

def save_metrics(metrics: Dict[str, float], save_path: str) -> None:
    """
    Lưu metrics vào file JSON
    
    Args:
        metrics: Dictionary chứa các metrics
        save_path: Đường dẫn để lưu file
    """
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)

def evaluate_trading_performance(predictions: np.ndarray, 
                              actual_prices: np.ndarray,
                              initial_balance: float = 10000.0) -> Dict[str, float]:
    """
    Đánh giá hiệu suất trading dựa trên predictions
    
    Args:
        predictions: Predicted price movements
        actual_prices: Actual price values
        initial_balance: Initial trading balance
        
    Returns:
        Dictionary chứa các metrics trading
    """
    balance = initial_balance
    position = 0
    trades = []
    returns = []
    
    for i in range(1, len(predictions)):
        pred_change = predictions[i] - actual_prices[i-1]
        actual_change = actual_prices[i] - actual_prices[i-1]
        
        # Mô phỏng trading
        if pred_change > 0 and position == 0:  # Buy signal
            position = balance / actual_prices[i]
            balance = 0
            trades.append(('buy', actual_prices[i]))
        elif pred_change < 0 and position > 0:  # Sell signal
            balance = position * actual_prices[i]
            returns.append((balance - initial_balance) / initial_balance)
            position = 0
            trades.append(('sell', actual_prices[i]))
    
    # Tính các metrics
    if len(returns) == 0:
        returns.append(0)
    
    final_balance = balance + (position * actual_prices[-1])
    total_return = (final_balance - initial_balance) / initial_balance
    sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0
    max_drawdown = calculate_max_drawdown(returns)
    
    return {
        'total_return': float(total_return),
        'sharpe_ratio': float(sharpe_ratio),
        'max_drawdown': float(max_drawdown),
        'number_of_trades': len(trades),
        'final_balance': float(final_balance)
    }

def calculate_max_drawdown(returns: List[float]) -> float:
    """
    Tính toán maximum drawdown từ list of returns
    
    Args:
        returns: List các giá trị return
        
    Returns:
        Maximum drawdown value
    """
    cumulative = np.cumprod(np.array(returns) + 1)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative / running_max - 1
    return float(np.min(drawdown))

def prepare_training_report(model_name: str,
                          training_history: Dict[str, List[float]],
                          metrics: Dict[str, float],
                          trading_metrics: Dict[str, float],
                          save_dir: str) -> None:
    """
    Tạo báo cáo training tổng hợp
    
    Args:
        model_name: Tên của model
        training_history: Lịch sử training
        metrics: Các metrics đánh giá model
        trading_metrics: Các metrics đánh giá trading
        save_dir: Thư mục lưu báo cáo
    """
    report = {
        'model_name': model_name,
        'training_metrics': metrics,
        'trading_performance': trading_metrics
    }
    
    # Lưu report
    report_path = os.path.join(save_dir, f'{model_name}_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    # Vẽ đồ thị training history
    plot_path = os.path.join(save_dir, f'{model_name}_training_history.png')
    plot_training_history(training_history, plot_path)
    
    logger.info(f"Training report saved to {report_path}")
    logger.info(f"Training history plot saved to {plot_path}")

def load_and_preprocess_data(data_dir: str, symbol: str, interval: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load và tiền xử lý dữ liệu cho training
    
    Args:
        data_dir: Thư mục chứa dữ liệu
        symbol: Trading pair symbol
        interval: Timeframe interval
        
    Returns:
        Tuple của (X, y) đã xử lý
    """
    try:
        X = np.load(os.path.join(data_dir, f'{symbol}_{interval}_X.npy'))
        y = np.load(os.path.join(data_dir, f'{symbol}_{interval}_y.npy'))
        
        # Kiểm tra dữ liệu
        if X.size == 0 or y.size == 0:
            raise ValueError("Empty data arrays")
        
        logger.info(f"Loaded data: X shape {X.shape}, y shape {y.shape}")
        return X, y
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return None, None

def load_model(model: torch.nn.Module, model_path: str) -> torch.nn.Module:
    """
    Load model từ file checkpoint
    
    Args:
        model: Model instance
        model_path: Đường dẫn đến file model
        
    Returns:
        Model đã load weights
    """
    try:
        model.load_state_dict(torch.load(model_path))
        model.eval()
        logger.info(f"Loaded model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {str(e)}")
        return None

def save_predictions(predictions: np.ndarray, symbol: str, interval: str) -> None:
    """
    Lưu predictions vào file
    
    Args:
        predictions: Numpy array của predictions
        symbol: Cặp giao dịch
        interval: Khung thời gian
    """
    try:
        filename = f"{symbol}_{interval}_predictions.npy"
        filepath = os.path.join(PATHS['data_dir'], filename)
        np.save(filepath, predictions)
        logger.info(f"Saved predictions to {filepath}")
    except Exception as e:
        logger.error(f"Error saving predictions: {str(e)}")

def generate_trading_signals(predictions: np.ndarray, threshold: float = 0.01) -> List[int]:
    """
    Tạo tín hiệu giao dịch từ predictions
    
    Args:
        predictions: Numpy array của predictions
        threshold: Ngưỡng để tạo tín hiệu (% thay đổi giá)
        
    Returns:
        List các tín hiệu (1: Buy, 0: Hold, -1: Sell)
    """
    try:
        signals = []
        for i in range(1, len(predictions)):
            price_change = (predictions[i] - predictions[i-1]) / predictions[i-1]
            
            if price_change > threshold:
                signals.append(1)  # Buy signal
            elif price_change < -threshold:
                signals.append(-1)  # Sell signal
            else:
                signals.append(0)  # Hold signal
        
        logger.info(f"Generated {len(signals)} trading signals")
        return signals
    except Exception as e:
        logger.error(f"Error generating trading signals: {str(e)}")
        return None

def backtest_strategy(prices: np.ndarray, signals: List[int], 
                     initial_balance: float = 10000.0) -> Tuple[float, List[float]]:
    """
    Backtest chiến lược giao dịch
    
    Args:
        prices: Numpy array của giá
        signals: List các tín hiệu giao dịch
        initial_balance: Số dư ban đầu
        
    Returns:
        Tuple của (final_balance, balance_history)
    """
    try:
        balance = initial_balance
        position = 0
        balance_history = [balance]
        
        for i in range(len(signals)):
            if signals[i] == 1 and position == 0:  # Buy signal
                position = balance / prices[i]
                balance = 0
            elif signals[i] == -1 and position > 0:  # Sell signal
                balance = position * prices[i]
                position = 0
            
            # Cập nhật balance history
            current_value = balance + (position * prices[i] if position > 0 else 0)
            balance_history.append(current_value)
        
        final_balance = balance + (position * prices[-1] if position > 0 else 0)
        returns = (final_balance - initial_balance) / initial_balance * 100
        
        logger.info(f"Backtest results: Initial={initial_balance}, Final={final_balance}, Returns={returns}%")
        return final_balance, balance_history
    except Exception as e:
        logger.error(f"Error in backtesting: {str(e)}")
        return None, None

def plot_results(prices: np.ndarray, predictions: np.ndarray, 
                balance_history: List[float], save_path: str) -> None:
    """
    Vẽ đồ thị kết quả
    
    Args:
        prices: Numpy array của giá
        predictions: Numpy array của predictions
        balance_history: List lịch sử số dư
        save_path: Đường dẫn để lưu plot
    """
    try:
        # Tạo figure với 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        
        # Plot 1: Giá thực tế vs Predictions
        ax1.plot(prices, label='Actual')
        ax1.plot(predictions, label='Predicted')
        ax1.set_title('Price vs Predictions')
        ax1.legend()
        
        # Plot 2: Prediction Error
        error = prices - predictions
        ax2.plot(error)
        ax2.set_title('Prediction Error')
        
        # Plot 3: Balance History
        ax3.plot(balance_history)
        ax3.set_title('Account Balance')
        
        # Lưu plot
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"Saved plots to {save_path}")
    except Exception as e:
        logger.error(f"Error plotting results: {str(e)}")

def prepare_live_data(data: pd.DataFrame, sequence_length: int = 60) -> torch.Tensor:
    """
    Chuẩn bị dữ liệu cho dự đoán real-time
    
    Args:
        data: DataFrame chứa dữ liệu live
        sequence_length: Độ dài chuỗi đầu vào
        
    Returns:
        Tensor dữ liệu đã chuẩn bị
    """
    try:
        # Chỉ lấy sequence_length mẫu cuối cùng
        if len(data) > sequence_length:
            data = data.tail(sequence_length)
        
        # Chuyển đổi thành numpy array
        data_array = data.values
        
        # Chuẩn hóa dữ liệu (sử dụng cùng scaler với training)
        scaler_path = os.path.join(PATHS['models_dir'], 'feature_scaler.pkl')
        if os.path.exists(scaler_path):
            import joblib
            scaler = joblib.load(scaler_path)
            data_array = scaler.transform(data_array)
        
        # Chuyển đổi thành tensor
        data_tensor = torch.FloatTensor(data_array).unsqueeze(0)  # Add batch dimension
        
        logger.info(f"Prepared live data with shape {data_tensor.shape}")
        return data_tensor
    except Exception as e:
        logger.error(f"Error preparing live data: {str(e)}")
        return None 