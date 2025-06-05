import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException
import logging
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRADING_PAIRS, TIMEFRAMES, DATA_CONFIG, PATHS

# Tạo thư mục logs nếu chưa tồn tại
os.makedirs(PATHS['logs_dir'], exist_ok=True)

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PATHS['logs_dir'], 'data_collection.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BinanceDataCollector:
    def __init__(self):
        """Khởi tạo Binance Data Collector"""
        load_dotenv()
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_SECRET_KEY')
        self.client = Client(self.api_key, self.api_secret)
        self.data_dir = PATHS['data_dir']
        os.makedirs(self.data_dir, exist_ok=True)

    def get_historical_klines(self, symbol: str, interval: str, start_date: datetime) -> pd.DataFrame:
        """
        Thu thập dữ liệu historical klines từ Binance
        
        Args:
            symbol: Cặp giao dịch (e.g., 'BTCUSDT')
            interval: Khung thời gian (e.g., '1h', '4h', '1d')
            start_date: Ngày bắt đầu thu thập
            
        Returns:
            DataFrame chứa dữ liệu OHLCV
        """
        try:
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_date.strftime('%Y-%m-%d'),
                limit=1000
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignored'
            ])
            
            # Chuyển đổi kiểu dữ liệu
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
                
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
        except BinanceAPIException as e:
            logger.error(f"Error collecting data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def get_orderbook_data(self, symbol: str, limit: int = 10) -> pd.DataFrame:
        """
        Thu thập dữ liệu orderbook từ Binance
        
        Args:
            symbol: Cặp giao dịch
            limit: Độ sâu của orderbook
            
        Returns:
            DataFrame chứa dữ liệu orderbook
        """
        try:
            depth = self.client.get_order_book(symbol=symbol, limit=limit)
            bids_df = pd.DataFrame(depth['bids'], columns=['price', 'quantity'])
            asks_df = pd.DataFrame(depth['asks'], columns=['price', 'quantity'])
            
            bids_df['side'] = 'bid'
            asks_df['side'] = 'ask'
            
            df = pd.concat([bids_df, asks_df])
            df['timestamp'] = pd.Timestamp.now()
            return df
            
        except BinanceAPIException as e:
            logger.error(f"Error collecting orderbook data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def collect_and_save_data(self):
        """Thu thập và lưu trữ dữ liệu cho tất cả cặp giao dịch và timeframes"""
        start_date = datetime.now() - timedelta(days=DATA_CONFIG['historical_data_days'])
        
        for symbol in TRADING_PAIRS:
            logger.info(f"Collecting data for {symbol}")
            
            # Thu thập dữ liệu OHLCV
            for interval in TIMEFRAMES:
                df = self.get_historical_klines(symbol, interval, start_date)
                if not df.empty:
                    filename = f"{symbol}_{interval}_klines.csv"
                    df.to_csv(os.path.join(self.data_dir, filename), index=False)
                    logger.info(f"Saved {filename}")
            
            # Thu thập dữ liệu orderbook
            df_orderbook = self.get_orderbook_data(symbol, DATA_CONFIG['orderbook_depth'])
            if not df_orderbook.empty:
                filename = f"{symbol}_orderbook.csv"
                df_orderbook.to_csv(os.path.join(self.data_dir, filename), index=False)
                logger.info(f"Saved {filename}")

def main():
    """Hàm main để chạy thu thập dữ liệu"""
    try:
        collector = BinanceDataCollector()
        collector.collect_and_save_data()
        logger.info("Data collection completed successfully")
    except Exception as e:
        logger.error(f"Error in data collection: {str(e)}")

if __name__ == "__main__":
    main() 