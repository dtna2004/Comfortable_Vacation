import os
import sys
import pandas as pd
import numpy as np
from typing import Tuple, List
import ta
from sklearn.preprocessing import MinMaxScaler
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRADING_PAIRS, TIMEFRAMES, DATA_CONFIG, PATHS

# Tạo thư mục logs nếu chưa tồn tại
os.makedirs(PATHS['logs_dir'], exist_ok=True)

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PATHS['logs_dir'], 'preprocessing.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        """Khởi tạo Data Preprocessor"""
        self.data_dir = PATHS['data_dir']
        self.scalers = {}
        
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Thêm các chỉ báo kỹ thuật vào DataFrame
        
        Args:
            df: DataFrame với dữ liệu OHLCV
            
        Returns:
            DataFrame với các chỉ báo kỹ thuật
        """
        # Trend Indicators
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
        df['macd'] = ta.trend.macd_diff(df['close'])
        
        # Momentum Indicators
        df['rsi'] = ta.momentum.rsi(df['close'])
        df['stoch'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
        df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'])
        
        # Volatility Indicators
        df['bb_high'] = ta.volatility.bollinger_hband(df['close'])
        df['bb_low'] = ta.volatility.bollinger_lband(df['close'])
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        
        # Volume Indicators
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        df['mfi'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'])
        
        return df
        
    def prepare_sequence_data(self, df: pd.DataFrame, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """
        Chuẩn bị dữ liệu dạng chuỗi cho model
        
        Args:
            df: DataFrame đã có các chỉ báo kỹ thuật
            sequence_length: Độ dài chuỗi đầu vào
            
        Returns:
            Tuple của (X, y) cho training
        """
        # Loại bỏ các hàng có giá trị NaN
        df = df.dropna()
        
        # Loại bỏ cột timestamp trước khi chuẩn hóa
        feature_columns = df.columns.drop(['timestamp'])
        df_features = df[feature_columns]
        
        # Chuẩn hóa dữ liệu
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df_features)
        self.scalers['feature_scaler'] = scaler
        
        # Tạo chuỗi dữ liệu
        X, y = [], []
        for i in range(len(scaled_data) - sequence_length):
            X.append(scaled_data[i:(i + sequence_length)])
            # Target là giá đóng cửa của ngày tiếp theo
            y.append(scaled_data[i + sequence_length, df_features.columns.get_loc('close')])
            
        return np.array(X), np.array(y)
        
    def process_orderbook_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Xử lý dữ liệu orderbook
        
        Args:
            df: DataFrame chứa dữ liệu orderbook
            
        Returns:
            DataFrame đã xử lý
        """
        # Chuyển đổi kiểu dữ liệu
        df['price'] = df['price'].astype(float)
        df['quantity'] = df['quantity'].astype(float)
        
        # Tính toán các features từ orderbook
        bid_data = df[df['side'] == 'bid']
        ask_data = df[df['side'] == 'ask']
        
        # Tính spread
        best_bid = bid_data['price'].max()
        best_ask = ask_data['price'].min()
        spread = best_ask - best_bid
        
        # Tính volume weighted average price (VWAP)
        vwap = (df['price'] * df['quantity']).sum() / df['quantity'].sum()
        
        # Tạo DataFrame mới với các features đã tính toán
        result = pd.DataFrame({
            'timestamp': [df['timestamp'].iloc[0]],
            'best_bid': [best_bid],
            'best_ask': [best_ask],
            'spread': [spread],
            'vwap': [vwap],
            'bid_volume': [bid_data['quantity'].sum()],
            'ask_volume': [ask_data['quantity'].sum()]
        })
        
        return result
        
    def prepare_data(self, symbol: str, interval: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Chuẩn bị dữ liệu cho một cặp giao dịch và timeframe cụ thể
        
        Args:
            symbol: Cặp giao dịch
            interval: Khung thời gian
            
        Returns:
            Tuple của (X, y) cho training
        """
        try:
            # Đọc dữ liệu OHLCV
            klines_file = os.path.join(self.data_dir, f"{symbol}_{interval}_klines.csv")
            df = pd.read_csv(klines_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            logger.info(f"Loaded {len(df)} rows from {klines_file}")
            
            # Thêm các chỉ báo kỹ thuật
            df = self.add_technical_indicators(df)
            logger.info(f"Added technical indicators, shape: {df.shape}")
            
            # Loại bỏ các hàng có giá trị NaN từ chỉ báo kỹ thuật
            df = df.dropna()
            logger.info(f"After dropping NaN values from technical indicators, shape: {df.shape}")
            
            # Đọc và xử lý dữ liệu orderbook (nếu có)
            orderbook_file = os.path.join(self.data_dir, f"{symbol}_orderbook.csv")
            if os.path.exists(orderbook_file):
                orderbook_df = pd.read_csv(orderbook_file)
                orderbook_df['timestamp'] = pd.to_datetime(orderbook_df['timestamp'])
                orderbook_features = self.process_orderbook_data(orderbook_df)
                logger.info(f"Processed orderbook data, shape: {orderbook_features.shape}")
                
                # Merge với dữ liệu OHLCV
                df = pd.merge_asof(
                    df.sort_values('timestamp'),
                    orderbook_features.sort_values('timestamp'),
                    on='timestamp',
                    direction='nearest'  # Sử dụng giá trị gần nhất
                )
                logger.info(f"Merged with orderbook data, shape: {df.shape}")
                
                # Fill missing orderbook values with forward fill
                orderbook_columns = ['best_bid', 'best_ask', 'spread', 'vwap', 'bid_volume', 'ask_volume']
                df[orderbook_columns] = df[orderbook_columns].fillna(method='ffill')
                df[orderbook_columns] = df[orderbook_columns].fillna(method='bfill')
            
            # Loại bỏ cột timestamp trước khi chuẩn hóa
            feature_columns = df.columns.drop(['timestamp'])
            df_features = df[feature_columns]
            logger.info(f"Feature columns: {list(feature_columns)}")
            
            # Chuẩn hóa dữ liệu
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df_features)
            self.scalers['feature_scaler'] = scaler
            logger.info(f"Scaled data shape: {scaled_data.shape}")
            
            # Tạo chuỗi dữ liệu
            sequence_length = 60  # Độ dài chuỗi đầu vào
            X, y = [], []
            for i in range(len(scaled_data) - sequence_length):
                X.append(scaled_data[i:(i + sequence_length)])
                # Target là giá đóng cửa của ngày tiếp theo
                y.append(scaled_data[i + sequence_length, df_features.columns.get_loc('close')])
            
            X = np.array(X)
            y = np.array(y)
            logger.info(f"Final X shape: {X.shape}, y shape: {y.shape}")
            
            # Lưu dữ liệu đã xử lý
            np.save(os.path.join(self.data_dir, f"{symbol}_{interval}_X.npy"), X)
            np.save(os.path.join(self.data_dir, f"{symbol}_{interval}_y.npy"), y)
            
            # Lưu scaler
            import joblib
            joblib.dump(scaler, os.path.join(PATHS['models_dir'], 'feature_scaler.pkl'))
            
            logger.info(f"Saved processed data for {symbol} {interval}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing data for {symbol} {interval}: {str(e)}")
            return None, None

def main():
    """Hàm main để chạy xử lý dữ liệu"""
    try:
        preprocessor = DataPreprocessor()
        for symbol in TRADING_PAIRS:
            for interval in TIMEFRAMES:
                X, y = preprocessor.prepare_data(symbol, interval)
                if X is not None and y is not None:
                    # Lưu dữ liệu đã xử lý
                    np.save(os.path.join(preprocessor.data_dir, f"{symbol}_{interval}_X.npy"), X)
                    np.save(os.path.join(preprocessor.data_dir, f"{symbol}_{interval}_y.npy"), y)
                    logger.info(f"Saved processed data for {symbol} {interval}")
                    
        logger.info("Data preprocessing completed successfully")
    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")

if __name__ == "__main__":
    main() 