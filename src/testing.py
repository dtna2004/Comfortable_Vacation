import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple
import json
import matplotlib.pyplot as plt
from binance.client import Client
from sklearn.model_selection import ParameterGrid
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRADING_PAIRS, TIMEFRAMES, PATHS
from model import TransformerLSTM
from utils import prepare_live_data, load_model
from risk_management import RiskManager

# Thiết lập logging
os.makedirs(PATHS['logs_dir'], exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PATHS['logs_dir'], 'testing.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingTester:
    def __init__(self):
        """Khởi tạo Trading Tester"""
        from dotenv import load_dotenv
        load_dotenv()
        
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_SECRET_KEY')
        self.client = Client(self.api_key, self.api_secret)
        
        # Tạo thư mục cho test results
        self.test_dir = os.path.join(PATHS['data_dir'], 'test_results')
        os.makedirs(self.test_dir, exist_ok=True)
        
    def get_historical_data(self, symbol: str, interval: str, 
                          start_date: datetime) -> pd.DataFrame:
        """
        Lấy dữ liệu lịch sử từ Binance
        
        Args:
            symbol: Trading pair
            interval: Timeframe
            start_date: Ngày bắt đầu
            
        Returns:
            DataFrame chứa dữ liệu lịch sử
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
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
                
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            logger.error(f"Error getting historical data: {str(e)}")
            return pd.DataFrame()
            
    def run_backtest(self, symbol: str, interval: str, 
                    start_date: datetime, initial_balance: float = 10000.0,
                    risk_params: Dict = None) -> Dict:
        """
        Chạy backtest với các tham số cho trước
        
        Args:
            symbol: Trading pair
            interval: Timeframe
            start_date: Ngày bắt đầu
            initial_balance: Số dư ban đầu
            risk_params: Tham số risk management
            
        Returns:
            Dictionary chứa kết quả backtest
        """
        try:
            # Lấy dữ liệu lịch sử
            df = self.get_historical_data(symbol, interval, start_date)
            if df.empty:
                return {}
                
            # Load model
            model = load_model(
                TransformerLSTM(...),  # TODO: Add model parameters
                os.path.join(PATHS['models_dir'], f"{symbol}_{interval}_model.pth")
            )
            
            if model is None:
                return {}
                
            # Khởi tạo risk manager với custom params nếu có
            risk_manager = RiskManager(initial_balance)
            if risk_params:
                risk_manager.max_position_size = risk_params.get('max_position_size', 0.1)
                risk_manager.stop_loss = risk_params.get('stop_loss', 0.02)
                risk_manager.take_profit = risk_params.get('take_profit', 0.05)
                
            # Biến theo dõi
            balance = initial_balance
            position = 0
            trades = []
            equity_curve = [balance]
            
            # Chạy backtest
            for i in range(60, len(df)):  # 60 là sequence length
                current_price = df['close'].iloc[i]
                
                # Chuẩn bị dữ liệu cho prediction
                data_window = df.iloc[i-60:i]
                X = prepare_live_data(data_window)
                
                if X is not None:
                    # Dự đoán
                    with torch.no_grad():
                        prediction = model(X).item()
                        
                    # Tính toán tín hiệu
                    price_change = (prediction - current_price) / current_price
                    
                    # Trading logic
                    if position == 0:  # Không có position
                        if price_change > risk_manager.take_profit:
                            # Tính position size
                            quantity = risk_manager.calculate_position_size(symbol, current_price)
                            cost = quantity * current_price
                            
                            if cost <= balance:
                                position = quantity
                                balance -= cost
                                trades.append({
                                    'timestamp': df['timestamp'].iloc[i],
                                    'type': 'BUY',
                                    'price': current_price,
                                    'quantity': quantity
                                })
                                
                    else:  # Đang có position
                        if (price_change < -risk_manager.stop_loss or 
                            price_change < -risk_manager.take_profit):
                            # Bán
                            revenue = position * current_price
                            balance += revenue
                            
                            trades.append({
                                'timestamp': df['timestamp'].iloc[i],
                                'type': 'SELL',
                                'price': current_price,
                                'quantity': position
                            })
                            
                            position = 0
                            
                # Cập nhật equity curve
                current_equity = balance + (position * current_price if position > 0 else 0)
                equity_curve.append(current_equity)
                
            # Tính các metrics
            final_balance = balance + (position * df['close'].iloc[-1] if position > 0 else 0)
            returns = (final_balance - initial_balance) / initial_balance
            
            if len(trades) > 0:
                trades_df = pd.DataFrame(trades)
                winning_trades = len(trades_df[trades_df['type'] == 'SELL'])
                
                # Tính PnL cho mỗi trade
                pnl = []
                for i in range(0, len(trades), 2):
                    if i + 1 < len(trades):
                        buy_trade = trades[i]
                        sell_trade = trades[i + 1]
                        trade_pnl = (sell_trade['price'] - buy_trade['price']) * buy_trade['quantity']
                        pnl.append(trade_pnl)
                        
                avg_profit = np.mean([p for p in pnl if p > 0]) if any(p > 0 for p in pnl) else 0
                avg_loss = abs(np.mean([p for p in pnl if p < 0])) if any(p < 0 for p in pnl) else 0
                max_drawdown = self.calculate_max_drawdown(equity_curve)
                
                results = {
                    'symbol': symbol,
                    'interval': interval,
                    'initial_balance': initial_balance,
                    'final_balance': final_balance,
                    'returns': returns,
                    'total_trades': len(trades) // 2,
                    'winning_trades': winning_trades,
                    'win_rate': winning_trades / (len(trades) // 2),
                    'avg_profit': avg_profit,
                    'avg_loss': avg_loss,
                    'profit_factor': avg_profit / avg_loss if avg_loss > 0 else float('inf'),
                    'max_drawdown': max_drawdown,
                    'risk_params': risk_params or {}
                }
                
                # Lưu kết quả
                self.save_test_results(results, equity_curve)
                
                return results
                
            return {}
            
        except Exception as e:
            logger.error(f"Error in backtest: {str(e)}")
            return {}
            
    def calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Tính maximum drawdown từ equity curve"""
        peaks = pd.Series(equity_curve).expanding(min_periods=1).max()
        drawdowns = (pd.Series(equity_curve) - peaks) / peaks
        return abs(float(drawdowns.min()))
        
    def save_test_results(self, results: Dict, equity_curve: List[float]):
        """Lưu kết quả test"""
        # Lưu results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(
            self.test_dir,
            f"backtest_results_{results['symbol']}_{results['interval']}_{timestamp}.json"
        )
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
            
        # Vẽ equity curve
        plt.figure(figsize=(12, 6))
        plt.plot(equity_curve)
        plt.title('Equity Curve')
        plt.xlabel('Trade Number')
        plt.ylabel('Account Value')
        plt.grid(True)
        
        plot_file = os.path.join(
            self.test_dir,
            f"equity_curve_{results['symbol']}_{results['interval']}_{timestamp}.png"
        )
        plt.savefig(plot_file)
        plt.close()
        
    def optimize_parameters(self, symbol: str, interval: str, 
                          start_date: datetime) -> Dict:
        """
        Tối ưu hóa các tham số giao dịch
        
        Args:
            symbol: Trading pair
            interval: Timeframe
            start_date: Ngày bắt đầu
            
        Returns:
            Dictionary chứa tham số tối ưu
        """
        # Grid search parameters
        param_grid = {
            'max_position_size': [0.05, 0.1, 0.15],
            'stop_loss': [0.01, 0.02, 0.03],
            'take_profit': [0.03, 0.05, 0.07]
        }
        
        best_return = -float('inf')
        best_params = None
        results = []
        
        # Grid search
        for params in ParameterGrid(param_grid):
            logger.info(f"Testing parameters: {params}")
            
            backtest_results = self.run_backtest(
                symbol=symbol,
                interval=interval,
                start_date=start_date,
                risk_params=params
            )
            
            if backtest_results and backtest_results['returns'] > best_return:
                best_return = backtest_results['returns']
                best_params = params
                
            results.append({
                'params': params,
                'results': backtest_results
            })
            
        # Lưu kết quả optimization
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        optim_file = os.path.join(
            self.test_dir,
            f"optimization_results_{symbol}_{interval}_{timestamp}.json"
        )
        
        with open(optim_file, 'w') as f:
            json.dump({
                'best_params': best_params,
                'best_return': best_return,
                'all_results': results
            }, f, indent=4)
            
        return best_params
        
def main():
    """Hàm main để chạy testing"""
    try:
        tester = TradingTester()
        
        # Test với mỗi cặp giao dịch
        for symbol in TRADING_PAIRS:
            for interval in TIMEFRAMES:
                logger.info(f"Testing {symbol} {interval}")
                
                # Chạy backtest
                start_date = datetime.now() - timedelta(days=30)  # Test 30 ngày
                results = tester.run_backtest(symbol, interval, start_date)
                
                if results:
                    logger.info(f"Backtest results for {symbol} {interval}:")
                    logger.info(json.dumps(results, indent=2))
                    
                    # Tối ưu tham số
                    best_params = tester.optimize_parameters(symbol, interval, start_date)
                    logger.info(f"Optimized parameters for {symbol} {interval}:")
                    logger.info(json.dumps(best_params, indent=2))
                    
        logger.info("Testing completed successfully")
        
    except Exception as e:
        logger.error(f"Error in testing: {str(e)}")

if __name__ == "__main__":
    main() 