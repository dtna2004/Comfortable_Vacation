import os
import sys
import argparse
from datetime import datetime
import logging
import json
import subprocess
from typing import List, Dict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRADING_PAIRS, TIMEFRAMES, PATHS
from data_collection import BinanceDataCollector
from preprocessing import DataPreprocessor
from model import TransformerLSTM
from training import ModelTrainer
from testing import TradingTester
from optimization import ModelOptimizer
from performance_analysis import PerformanceAnalyzer

# Thiết lập logging
os.makedirs(PATHS['logs_dir'], exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PATHS['logs_dir'], 'main.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingBotManager:
    def __init__(self):
        """Khởi tạo Trading Bot Manager"""
        self.data_collector = BinanceDataCollector()
        self.preprocessor = DataPreprocessor()
        self.trainer = ModelTrainer()
        self.tester = TradingTester()
        self.optimizer = ModelOptimizer()
        self.analyzer = PerformanceAnalyzer()
        
    def setup_environment(self, force_setup: bool = False):
        """
        Bước 1: Thiết lập môi trường
        
        Args:
            force_setup: Có cài đặt lại môi trường không
        """
        try:
            logger.info("Setting up environment...")
            
            # Kiểm tra và tạo các thư mục cần thiết
            for path in PATHS.values():
                os.makedirs(path, exist_ok=True)
                
            # Kiểm tra file .env
            if not os.path.exists('.env') or force_setup:
                api_key = input("Enter your Binance API key: ")
                api_secret = input("Enter your Binance API secret: ")
                
                with open('.env', 'w') as f:
                    f.write(f"BINANCE_API_KEY={api_key}\n")
                    f.write(f"BINANCE_SECRET_KEY={api_secret}\n")
                    
            # Cài đặt dependencies
            if force_setup:
                subprocess.run(['pip', 'install', '-r', 'requirements.txt'])
                
            logger.info("Environment setup completed")
            return True
            
        except Exception as e:
            logger.error(f"Error in environment setup: {str(e)}")
            return False
            
    def collect_data(self, symbols: List[str] = None, 
                    timeframes: List[str] = None,
                    days: int = 30):
        """
        Bước 2: Thu thập dữ liệu
        
        Args:
            symbols: Danh sách cặp giao dịch
            timeframes: Danh sách khung thời gian
            days: Số ngày dữ liệu
        """
        try:
            logger.info("Collecting data...")
            symbols = symbols or TRADING_PAIRS
            timeframes = timeframes or TIMEFRAMES
            
            for symbol in symbols:
                for timeframe in timeframes:
                    logger.info(f"Collecting {symbol} {timeframe} data...")
                    self.data_collector.collect_historical_data(
                        symbol=symbol,
                        interval=timeframe,
                        days=days
                    )
                    
            logger.info("Data collection completed")
            return True
            
        except Exception as e:
            logger.error(f"Error in data collection: {str(e)}")
            return False
            
    def build_model(self, symbols: List[str] = None,
                   timeframes: List[str] = None):
        """
        Bước 3: Xây dựng model
        
        Args:
            symbols: Danh sách cặp giao dịch
            timeframes: Danh sách khung thời gian
        """
        try:
            logger.info("Building models...")
            symbols = symbols or TRADING_PAIRS
            timeframes = timeframes or TIMEFRAMES
            
            for symbol in symbols:
                for timeframe in timeframes:
                    logger.info(f"Processing {symbol} {timeframe} data...")
                    
                    # Tiền xử lý dữ liệu
                    self.preprocessor.process_data(symbol, timeframe)
                    
                    # Khởi tạo model
                    model = TransformerLSTM(...)  # TODO: Add model parameters
                    
                    # Lưu model
                    model_path = os.path.join(
                        PATHS['models_dir'],
                        f"{symbol}_{timeframe}_model.pth"
                    )
                    torch.save(model.state_dict(), model_path)
                    
            logger.info("Model building completed")
            return True
            
        except Exception as e:
            logger.error(f"Error in model building: {str(e)}")
            return False
            
    def train_model(self, symbols: List[str] = None,
                    timeframes: List[str] = None,
                    epochs: int = 100):
        """
        Bước 4: Training pipeline
        
        Args:
            symbols: Danh sách cặp giao dịch
            timeframes: Danh sách khung thời gian
            epochs: Số epochs training
        """
        try:
            logger.info("Training models...")
            symbols = symbols or TRADING_PAIRS
            timeframes = timeframes or TIMEFRAMES
            
            for symbol in symbols:
                for timeframe in timeframes:
                    logger.info(f"Training {symbol} {timeframe} model...")
                    self.trainer.train(
                        symbol=symbol,
                        interval=timeframe,
                        epochs=epochs
                    )
                    
            logger.info("Model training completed")
            return True
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            return False
            
    def test_model(self, symbols: List[str] = None,
                   timeframes: List[str] = None):
        """
        Bước 5: Testing và tối ưu
        
        Args:
            symbols: Danh sách cặp giao dịch
            timeframes: Danh sách khung thời gian
        """
        try:
            logger.info("Testing models...")
            symbols = symbols or TRADING_PAIRS
            timeframes = timeframes or TIMEFRAMES
            
            for symbol in symbols:
                for timeframe in timeframes:
                    logger.info(f"Testing {symbol} {timeframe} model...")
                    
                    # Optimize hyperparameters
                    best_params = self.optimizer.optimize_hyperparameters(
                        symbol=symbol,
                        interval=timeframe
                    )
                    
                    # Run backtest
                    results = self.tester.run_backtest(
                        symbol=symbol,
                        interval=timeframe,
                        start_date=datetime.now(),
                        risk_params=best_params
                    )
                    
                    # Generate analysis report
                    report = self.analyzer.generate_analysis_report(
                        symbol=symbol,
                        interval=timeframe
                    )
                    
            logger.info("Model testing completed")
            return True
            
        except Exception as e:
            logger.error(f"Error in model testing: {str(e)}")
            return False
            
    def start_bot(self, symbols: List[str] = None):
        """
        Bước 6: Khởi động bot
        
        Args:
            symbols: Danh sách cặp giao dịch để theo dõi
        """
        try:
            logger.info("Starting trading bot dashboard...")
            symbols = symbols or ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
            
            # Export danh sách symbol để dashboard sử dụng
            with open(os.path.join(PATHS['data_dir'], 'monitor_pairs.json'), 'w') as f:
                json.dump(symbols, f)
                
            # Chạy dashboard
            subprocess.Popen(['streamlit', 'run', 'dashboard.py'])
            
            logger.info("Trading bot started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting bot: {str(e)}")
            return False

def main():
    """Hàm main với command line interface"""
    parser = argparse.ArgumentParser(description='Trading Bot Manager')
    
    parser.add_argument(
        'step',
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        help='Step to execute (1-6)'
    )
    
    parser.add_argument(
        '--symbols',
        nargs='+',
        help='Trading pairs to process'
    )
    
    parser.add_argument(
        '--timeframes',
        nargs='+',
        help='Timeframes to process'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Number of days for historical data'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--force-setup',
        action='store_true',
        help='Force environment setup'
    )
    
    args = parser.parse_args()
    
    try:
        manager = TradingBotManager()
        
        if args.step == 1:
            success = manager.setup_environment(args.force_setup)
        elif args.step == 2:
            success = manager.collect_data(args.symbols, args.timeframes, args.days)
        elif args.step == 3:
            success = manager.build_model(args.symbols, args.timeframes)
        elif args.step == 4:
            success = manager.train_model(args.symbols, args.timeframes, args.epochs)
        elif args.step == 5:
            success = manager.test_model(args.symbols, args.timeframes)
        elif args.step == 6:
            success = manager.start_bot(args.symbols)
            
        if success:
            logger.info(f"Step {args.step} completed successfully")
        else:
            logger.error(f"Step {args.step} failed")
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        
if __name__ == "__main__":
    main() 