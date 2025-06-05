import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List
import json
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PATHS

# Thiết lập logging
os.makedirs(PATHS['logs_dir'], exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PATHS['logs_dir'], 'monitoring.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingMonitor:
    def __init__(self):
        """Khởi tạo Trading Monitor"""
        self.trades_history = []
        self.performance_metrics = {}
        self.alerts = []
        
        # Tạo thư mục cho monitoring data
        self.monitor_dir = os.path.join(PATHS['data_dir'], 'monitoring')
        os.makedirs(self.monitor_dir, exist_ok=True)
        
    def log_trade(self, trade_info: Dict):
        """
        Ghi log thông tin giao dịch
        
        Args:
            trade_info: Dictionary chứa thông tin giao dịch
        """
        trade_info['timestamp'] = datetime.now()
        self.trades_history.append(trade_info)
        
        # Lưu vào file CSV
        df = pd.DataFrame([trade_info])
        trades_file = os.path.join(self.monitor_dir, 'trades_history.csv')
        
        if os.path.exists(trades_file):
            df.to_csv(trades_file, mode='a', header=False, index=False)
        else:
            df.to_csv(trades_file, index=False)
            
        logger.info(f"Logged trade: {trade_info}")
        
    def update_performance_metrics(self, metrics: Dict):
        """
        Cập nhật metrics hiệu suất
        
        Args:
            metrics: Dictionary chứa các metrics
        """
        self.performance_metrics.update(metrics)
        
        # Lưu metrics
        metrics_file = os.path.join(self.monitor_dir, 'performance_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(self.performance_metrics, f, indent=4)
            
        logger.info(f"Updated performance metrics: {metrics}")
        
    def generate_performance_report(self) -> Dict:
        """
        Tạo báo cáo hiệu suất tổng hợp
        
        Returns:
            Dictionary chứa báo cáo hiệu suất
        """
        if not self.trades_history:
            return {}
            
        trades_df = pd.DataFrame(self.trades_history)
        
        # Tính toán các metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_profit = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if losing_trades > 0 else 0
        
        profit_factor = avg_profit / avg_loss if avg_loss > 0 else float('inf')
        
        report = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_pnl': trades_df['pnl'].sum(),
            'max_drawdown': self.performance_metrics.get('max_drawdown', 0)
        }
        
        # Lưu báo cáo
        report_file = os.path.join(self.monitor_dir, 'performance_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=4)
            
        return report
        
    def plot_performance_charts(self):
        """Vẽ các biểu đồ hiệu suất"""
        if not self.trades_history:
            return
            
        trades_df = pd.DataFrame(self.trades_history)
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        trades_df.set_index('timestamp', inplace=True)
        
        # Tạo figure với 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        
        # Plot 1: Cumulative PnL
        cumulative_pnl = trades_df['pnl'].cumsum()
        ax1.plot(cumulative_pnl.index, cumulative_pnl.values)
        ax1.set_title('Cumulative PnL')
        ax1.grid(True)
        
        # Plot 2: Trade PnL Distribution
        trades_df['pnl'].hist(ax=ax2, bins=50)
        ax2.set_title('PnL Distribution')
        ax2.grid(True)
        
        # Plot 3: Win Rate Over Time
        trades_df['win'] = trades_df['pnl'] > 0
        win_rate = trades_df['win'].rolling(window=20).mean()
        ax3.plot(win_rate.index, win_rate.values)
        ax3.set_title('Win Rate (20-trade Moving Average)')
        ax3.grid(True)
        
        # Lưu plots
        plt.tight_layout()
        plt.savefig(os.path.join(self.monitor_dir, 'performance_charts.png'))
        plt.close()
        
    def add_alert(self, alert_type: str, message: str, severity: str = 'info'):
        """
        Thêm cảnh báo mới
        
        Args:
            alert_type: Loại cảnh báo
            message: Nội dung cảnh báo
            severity: Mức độ nghiêm trọng
        """
        alert = {
            'timestamp': datetime.now(),
            'type': alert_type,
            'message': message,
            'severity': severity
        }
        
        self.alerts.append(alert)
        
        # Log alert
        if severity == 'warning':
            logger.warning(f"{alert_type}: {message}")
        elif severity == 'error':
            logger.error(f"{alert_type}: {message}")
        else:
            logger.info(f"{alert_type}: {message}")
            
        # Lưu alerts
        alerts_file = os.path.join(self.monitor_dir, 'alerts.json')
        with open(alerts_file, 'w') as f:
            json.dump(self.alerts, f, indent=4)
            
    def check_system_health(self) -> bool:
        """
        Kiểm tra tình trạng hệ thống
        
        Returns:
            True nếu hệ thống hoạt động bình thường
        """
        try:
            # Kiểm tra disk space
            monitor_dir_size = sum(os.path.getsize(os.path.join(self.monitor_dir, f)) 
                                 for f in os.listdir(self.monitor_dir))
            if monitor_dir_size > 1e9:  # 1GB
                self.add_alert('disk_space', 'Monitor directory size exceeds 1GB', 'warning')
                
            # Kiểm tra log file size
            log_file = os.path.join(PATHS['logs_dir'], 'monitoring.log')
            if os.path.exists(log_file) and os.path.getsize(log_file) > 100e6:  # 100MB
                self.add_alert('log_size', 'Log file size exceeds 100MB', 'warning')
                
            # Kiểm tra số lượng alerts
            if len(self.alerts) > 1000:
                self.add_alert('alerts_count', 'Too many alerts in memory', 'warning')
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking system health: {str(e)}")
            return False
            
    def cleanup_old_data(self, days_to_keep: int = 30):
        """
        Xóa dữ liệu cũ
        
        Args:
            days_to_keep: Số ngày giữ lại dữ liệu
        """
        try:
            # Đọc trades history
            trades_file = os.path.join(self.monitor_dir, 'trades_history.csv')
            if os.path.exists(trades_file):
                df = pd.read_csv(trades_file)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Lọc theo ngày
                cutoff_date = datetime.now() - pd.Timedelta(days=days_to_keep)
                df = df[df['timestamp'] > cutoff_date]
                
                # Lưu lại
                df.to_csv(trades_file, index=False)
                
            # Xóa alerts cũ
            self.alerts = [alert for alert in self.alerts 
                         if alert['timestamp'] > cutoff_date]
                         
            alerts_file = os.path.join(self.monitor_dir, 'alerts.json')
            with open(alerts_file, 'w') as f:
                json.dump(self.alerts, f, indent=4)
                
            logger.info(f"Cleaned up data older than {days_to_keep} days")
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {str(e)}")
            
def main():
    """Hàm main để test monitoring"""
    try:
        monitor = TradingMonitor()
        
        # Test các functions
        monitor.log_trade({
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'price': 50000,
            'quantity': 0.1,
            'pnl': 100
        })
        
        monitor.update_performance_metrics({
            'win_rate': 0.6,
            'profit_factor': 1.5,
            'max_drawdown': 1000
        })
        
        monitor.generate_performance_report()
        monitor.plot_performance_charts()
        monitor.check_system_health()
        
        logger.info("Monitoring test completed successfully")
        
    except Exception as e:
        logger.error(f"Error in monitoring test: {str(e)}")

if __name__ == "__main__":
    main() 