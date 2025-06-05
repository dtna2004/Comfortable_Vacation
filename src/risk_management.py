import os
import sys
import numpy as np
from typing import Dict, List, Tuple
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PATHS

# Thiết lập logging
os.makedirs(PATHS['logs_dir'], exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PATHS['logs_dir'], 'risk_management.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self, initial_balance: float):
        """
        Khởi tạo Risk Manager
        
        Args:
            initial_balance: Số dư ban đầu
        """
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.positions = {}  # symbol -> position info
        self.risk_metrics = {
            'max_drawdown': 0,
            'current_drawdown': 0,
            'total_exposure': 0,
            'win_rate': 0,
            'profit_factor': 0
        }
        
        # Load risk parameters từ environment
        from dotenv import load_dotenv
        load_dotenv()
        
        self.max_position_size = float(os.getenv('MAX_POSITION_SIZE', '0.1'))  # 10% of balance
        self.max_total_exposure = float(os.getenv('MAX_TOTAL_EXPOSURE', '0.5'))  # 50% of balance
        self.stop_loss = float(os.getenv('STOP_LOSS', '0.02'))  # 2%
        self.take_profit = float(os.getenv('TAKE_PROFIT', '0.05'))  # 5%
        
    def calculate_position_size(self, symbol: str, price: float) -> float:
        """
        Tính toán kích thước position tối đa cho phép
        
        Args:
            symbol: Trading pair
            price: Giá hiện tại
            
        Returns:
            Kích thước position được phép
        """
        # Tính toán position size dựa trên % của balance
        max_position_value = self.current_balance * self.max_position_size
        
        # Kiểm tra tổng exposure
        current_exposure = sum(pos['value'] for pos in self.positions.values())
        remaining_exposure = self.current_balance * self.max_total_exposure - current_exposure
        
        # Lấy giá trị nhỏ nhất giữa max_position_value và remaining_exposure
        allowed_value = min(max_position_value, remaining_exposure)
        
        # Chuyển đổi sang số lượng
        quantity = allowed_value / price
        
        logger.info(f"Calculated position size for {symbol}: {quantity}")
        return quantity
        
    def check_stop_loss(self, symbol: str, current_price: float) -> bool:
        """
        Kiểm tra điều kiện stop loss
        
        Args:
            symbol: Trading pair
            current_price: Giá hiện tại
            
        Returns:
            True nếu cần stop loss
        """
        if symbol in self.positions:
            position = self.positions[symbol]
            price_change = (current_price - position['entry_price']) / position['entry_price']
            
            if price_change <= -self.stop_loss:
                logger.warning(f"Stop loss triggered for {symbol} at {current_price}")
                return True
        return False
        
    def check_take_profit(self, symbol: str, current_price: float) -> bool:
        """
        Kiểm tra điều kiện take profit
        
        Args:
            symbol: Trading pair
            current_price: Giá hiện tại
            
        Returns:
            True nếu đạt take profit
        """
        if symbol in self.positions:
            position = self.positions[symbol]
            price_change = (current_price - position['entry_price']) / position['entry_price']
            
            if price_change >= self.take_profit:
                logger.info(f"Take profit triggered for {symbol} at {current_price}")
                return True
        return False
        
    def update_position(self, symbol: str, quantity: float, price: float, side: str):
        """
        Cập nhật thông tin position
        
        Args:
            symbol: Trading pair
            quantity: Số lượng
            price: Giá
            side: 'BUY' hoặc 'SELL'
        """
        if side == 'BUY':
            self.positions[symbol] = {
                'quantity': quantity,
                'entry_price': price,
                'value': quantity * price
            }
        else:  # SELL
            if symbol in self.positions:
                # Tính P&L
                entry_value = self.positions[symbol]['quantity'] * self.positions[symbol]['entry_price']
                exit_value = quantity * price
                pnl = exit_value - entry_value
                
                # Cập nhật balance và metrics
                self.current_balance += pnl
                self.update_metrics(pnl)
                
                # Xóa position
                del self.positions[symbol]
                
    def update_metrics(self, pnl: float):
        """
        Cập nhật risk metrics sau mỗi giao dịch
        
        Args:
            pnl: Profit/Loss của giao dịch
        """
        # Cập nhật drawdown
        if pnl < 0:
            self.risk_metrics['current_drawdown'] += abs(pnl)
            if self.risk_metrics['current_drawdown'] > self.risk_metrics['max_drawdown']:
                self.risk_metrics['max_drawdown'] = self.risk_metrics['current_drawdown']
        else:
            self.risk_metrics['current_drawdown'] = 0
            
        # Cập nhật tổng exposure
        self.risk_metrics['total_exposure'] = sum(pos['value'] for pos in self.positions.values())
        
        logger.info(f"Updated risk metrics: {self.risk_metrics}")
        
    def check_risk_limits(self) -> bool:
        """
        Kiểm tra các giới hạn rủi ro
        
        Returns:
            True nếu trong giới hạn cho phép
        """
        # Kiểm tra drawdown
        if self.risk_metrics['current_drawdown'] > self.initial_balance * 0.1:  # 10% max drawdown
            logger.warning("Maximum drawdown limit reached")
            return False
            
        # Kiểm tra tổng exposure
        if self.risk_metrics['total_exposure'] > self.current_balance * self.max_total_exposure:
            logger.warning("Maximum total exposure limit reached")
            return False
            
        return True
        
    def get_risk_report(self) -> Dict:
        """
        Tạo báo cáo rủi ro
        
        Returns:
            Dictionary chứa các metrics rủi ro
        """
        return {
            'initial_balance': self.initial_balance,
            'current_balance': self.current_balance,
            'total_pnl': self.current_balance - self.initial_balance,
            'total_pnl_percent': ((self.current_balance - self.initial_balance) / self.initial_balance) * 100,
            'current_positions': len(self.positions),
            'total_exposure': self.risk_metrics['total_exposure'],
            'exposure_percent': (self.risk_metrics['total_exposure'] / self.current_balance) * 100,
            'max_drawdown': self.risk_metrics['max_drawdown'],
            'max_drawdown_percent': (self.risk_metrics['max_drawdown'] / self.initial_balance) * 100
        } 