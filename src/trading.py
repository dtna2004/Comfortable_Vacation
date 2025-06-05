import os
import sys
import numpy as np
import pandas as pd
import torch
from binance.client import Client
from binance.exceptions import BinanceAPIException
import logging
from datetime import datetime
from typing import Dict, List, Tuple
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRADING_PAIRS, TIMEFRAMES, PATHS
from model import TransformerLSTM
from utils import prepare_live_data, load_model

# Thiết lập logging
os.makedirs(PATHS['logs_dir'], exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PATHS['logs_dir'], 'trading.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self, symbol: str, interval: str):
        """
        Khởi tạo Trading Bot
        
        Args:
            symbol: Cặp giao dịch (e.g., 'BTCUSDT')
            interval: Khung thời gian (e.g., '1h')
        """
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        # Khởi tạo Binance client
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_SECRET_KEY')
        self.client = Client(self.api_key, self.api_secret)
        
        # Trading parameters
        self.symbol = symbol
        self.interval = interval
        self.position = 0
        self.balance = 0
        self.trades = []
        
        # Risk management parameters
        self.max_position_size = float(os.getenv('MAX_POSITION_SIZE', '0.1'))  # 10% of balance
        self.stop_loss = float(os.getenv('STOP_LOSS', '0.02'))  # 2%
        self.take_profit = float(os.getenv('TAKE_PROFIT', '0.05'))  # 5%
        
        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_trading_model()
        
        # Initialize metrics tracking
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0,
            'max_drawdown': 0
        }
        
    def _load_trading_model(self) -> TransformerLSTM:
        """Load và return pre-trained model"""
        try:
            model_path = os.path.join(PATHS['models_dir'], f"{self.symbol}_{self.interval}_model.pth")
            model = load_model(TransformerLSTM(...), model_path)  # TODO: Add model parameters
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None
            
    def get_account_balance(self) -> float:
        """Lấy số dư tài khoản"""
        try:
            account = self.client.get_account()
            return float(account['balances'][0]['free'])
        except BinanceAPIException as e:
            logger.error(f"Error getting account balance: {str(e)}")
            return 0
            
    def get_current_price(self) -> float:
        """Lấy giá hiện tại của symbol"""
        try:
            ticker = self.client.get_symbol_ticker(symbol=self.symbol)
            return float(ticker['price'])
        except BinanceAPIException as e:
            logger.error(f"Error getting current price: {str(e)}")
            return 0
            
    def calculate_position_size(self, price: float) -> float:
        """Tính toán kích thước position dựa trên risk management"""
        account_balance = self.get_account_balance()
        position_size = account_balance * self.max_position_size
        return position_size / price
        
    def place_order(self, side: str, quantity: float, price: float) -> bool:
        """
        Đặt lệnh giao dịch
        
        Args:
            side: 'BUY' hoặc 'SELL'
            quantity: Số lượng
            price: Giá
        """
        try:
            order = self.client.create_order(
                symbol=self.symbol,
                side=side,
                type='LIMIT',
                timeInForce='GTC',
                quantity=quantity,
                price=price
            )
            logger.info(f"Placed {side} order: {order}")
            return True
        except BinanceAPIException as e:
            logger.error(f"Error placing order: {str(e)}")
            return False
            
    def update_metrics(self, trade_result: float):
        """Cập nhật metrics sau mỗi giao dịch"""
        self.metrics['total_trades'] += 1
        if trade_result > 0:
            self.metrics['winning_trades'] += 1
        else:
            self.metrics['losing_trades'] += 1
        self.metrics['total_profit'] += trade_result
        
        # Cập nhật max drawdown
        if trade_result < self.metrics['max_drawdown']:
            self.metrics['max_drawdown'] = trade_result
            
    def predict_next_price(self, current_data: pd.DataFrame) -> float:
        """Dự đoán giá tiếp theo từ model"""
        try:
            # Chuẩn bị dữ liệu
            X = prepare_live_data(current_data)
            if X is None:
                return None
                
            # Dự đoán
            with torch.no_grad():
                X = X.to(self.device)
                prediction = self.model(X)
                return prediction.item()
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return None
            
    def execute_trade_signal(self, prediction: float, current_price: float):
        """
        Thực thi tín hiệu giao dịch dựa trên prediction
        
        Args:
            prediction: Giá dự đoán
            current_price: Giá hiện tại
        """
        price_change = (prediction - current_price) / current_price
        
        # Nếu đang không có position
        if self.position == 0:
            if price_change > self.take_profit:  # Tín hiệu mua
                quantity = self.calculate_position_size(current_price)
                if self.place_order('BUY', quantity, current_price):
                    self.position = quantity
                    self.trades.append(('BUY', current_price, quantity))
                    
        # Nếu đang có position
        else:
            if price_change < -self.stop_loss:  # Stop loss
                if self.place_order('SELL', self.position, current_price):
                    trade_result = (current_price - self.trades[-1][1]) * self.position
                    self.update_metrics(trade_result)
                    self.position = 0
            elif price_change < -self.take_profit:  # Take profit
                if self.place_order('SELL', self.position, current_price):
                    trade_result = (current_price - self.trades[-1][1]) * self.position
                    self.update_metrics(trade_result)
                    self.position = 0
                    
    def run(self, interval_seconds: int = 60):
        """
        Chạy trading bot
        
        Args:
            interval_seconds: Thời gian giữa các lần check (giây)
        """
        logger.info(f"Starting trading bot for {self.symbol}")
        
        while True:
            try:
                # Lấy dữ liệu hiện tại
                current_price = self.get_current_price()
                if current_price == 0:
                    continue
                    
                # TODO: Lấy và xử lý dữ liệu real-time cho prediction
                
                # Dự đoán giá tiếp theo
                prediction = self.predict_next_price(None)  # TODO: Add current data
                if prediction is not None:
                    # Thực thi giao dịch
                    self.execute_trade_signal(prediction, current_price)
                    
                # Log metrics
                logger.info(f"Current metrics: {self.metrics}")
                
                # Đợi đến interval tiếp theo
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                time.sleep(interval_seconds)
                
def main():
    """Hàm main để chạy trading bot"""
    try:
        # Khởi tạo trading bot cho từng cặp giao dịch
        bots = []
        for symbol in TRADING_PAIRS:
            for interval in TIMEFRAMES:
                bot = TradingBot(symbol, interval)
                bots.append(bot)
                
        # Chạy các bots
        for bot in bots:
            bot.run()
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main() 