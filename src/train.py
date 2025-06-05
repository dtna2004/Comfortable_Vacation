import os
import sys
import argparse
import logging
import torch
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRADING_PAIRS, TIMEFRAMES, MODEL_CONFIG
from training import ModelTrainer
from utils import (
    load_and_preprocess_data,
    calculate_metrics,
    evaluate_trading_performance,
    prepare_training_report
)

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/train.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train trading models')
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                      help='Trading pair symbol')
    parser.add_argument('--interval', type=str, default='1h',
                      help='Trading interval')
    parser.add_argument('--model', type=str, default='both',
                      choices=['transformer_lstm', 'rl', 'both'],
                      help='Model to train')
    parser.add_argument('--output_dir', type=str, default='../models',
                      help='Directory to save models and reports')
    return parser.parse_args()

def train_transformer_lstm(trainer: ModelTrainer, 
                         X: torch.Tensor, 
                         y: torch.Tensor,
                         output_dir: str) -> None:
    """
    Train và đánh giá Transformer-LSTM model
    
    Args:
        trainer: ModelTrainer instance
        X: Input features
        y: Target values
        output_dir: Directory to save outputs
    """
    # Chuẩn bị data loaders
    train_loader, val_loader = trainer.prepare_data(X, y)
    
    # Training
    logger.info("Starting Transformer-LSTM training...")
    history = trainer.train_transformer_lstm(
        train_loader, val_loader, input_dim=X.shape[2]
    )
    
    # Đánh giá model
    model = trainer.load_best_model(X.shape[2])
    predictions = trainer.predict(model, val_loader)
    
    # Tính toán metrics
    metrics = calculate_metrics(y[-len(predictions):], predictions)
    trading_metrics = evaluate_trading_performance(
        predictions, y[-len(predictions):].numpy()
    )
    
    # Tạo báo cáo
    prepare_training_report(
        'transformer_lstm',
        history,
        metrics,
        trading_metrics,
        output_dir
    )
    
    logger.info("Transformer-LSTM training and evaluation completed")

def train_rl_model(trainer: ModelTrainer,
                  X: torch.Tensor,
                  y: torch.Tensor,
                  output_dir: str) -> None:
    """
    Train và đánh giá RL model
    
    Args:
        trainer: ModelTrainer instance
        X: Input features
        y: Target values
        output_dir: Directory to save outputs
    """
    # Tạo environment
    env = trainer.create_trading_environment(X.reshape(-1, X.shape[2]))
    
    # Training
    logger.info("Starting RL model training...")
    trainer.train_rl_model(env)
    
    # Đánh giá model
    actions, rewards = trainer.evaluate_rl_model(env)
    
    # Tính toán metrics
    trading_metrics = {
        'total_reward': float(sum(rewards)),
        'average_reward': float(sum(rewards) / len(rewards)),
        'number_of_trades': len([a for a in actions if a != 1])  # Không tính HOLD
    }
    
    # Tạo báo cáo
    prepare_training_report(
        'rl_model',
        {'rewards': rewards},
        {},  # RL không có metrics truyền thống
        trading_metrics,
        output_dir
    )
    
    logger.info("RL model training and evaluation completed")

def main():
    """Main training pipeline"""
    args = parse_arguments()
    
    try:
        # Tạo output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Load và tiền xử lý dữ liệu
        X, y = load_and_preprocess_data('../data', args.symbol, args.interval)
        if X is None or y is None:
            raise ValueError("Failed to load data")
            
        # Khởi tạo trainer
        trainer = ModelTrainer(save_dir=args.output_dir)
        
        # Training theo lựa chọn model
        if args.model in ['transformer_lstm', 'both']:
            train_transformer_lstm(trainer, X, y, args.output_dir)
            
        if args.model in ['rl', 'both']:
            train_rl_model(trainer, X, y, args.output_dir)
            
        logger.info("Training pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 