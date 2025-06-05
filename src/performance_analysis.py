import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List, Tuple
import json
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRADING_PAIRS, TIMEFRAMES, PATHS

# Thiết lập logging
os.makedirs(PATHS['logs_dir'], exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PATHS['logs_dir'], 'performance_analysis.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    def __init__(self):
        """Khởi tạo Performance Analyzer"""
        # Tạo thư mục cho analysis results
        self.analysis_dir = os.path.join(PATHS['data_dir'], 'analysis_results')
        os.makedirs(self.analysis_dir, exist_ok=True)
        
    def load_test_results(self, symbol: str, interval: str) -> pd.DataFrame:
        """
        Load kết quả test từ file
        
        Args:
            symbol: Trading pair
            interval: Timeframe
            
        Returns:
            DataFrame chứa kết quả test
        """
        try:
            # Tìm file test results mới nhất
            test_files = [f for f in os.listdir(PATHS['data_dir']) 
                         if f.startswith(f"backtest_results_{symbol}_{interval}")]
            
            if not test_files:
                return pd.DataFrame()
                
            latest_file = sorted(test_files)[-1]
            with open(os.path.join(PATHS['data_dir'], latest_file)) as f:
                results = json.load(f)
                
            return pd.DataFrame([results])
            
        except Exception as e:
            logger.error(f"Error loading test results: {str(e)}")
            return pd.DataFrame()
            
    def analyze_trading_performance(self, results_df: pd.DataFrame) -> Dict:
        """
        Phân tích hiệu suất giao dịch
        
        Args:
            results_df: DataFrame chứa kết quả test
            
        Returns:
            Dictionary chứa các metrics phân tích
        """
        if results_df.empty:
            return {}
            
        try:
            # Tính các metrics
            total_return = results_df['returns'].mean()
            sharpe_ratio = results_df['returns'].mean() / results_df['returns'].std() \
                if results_df['returns'].std() != 0 else 0
            max_drawdown = results_df['max_drawdown'].max()
            win_rate = results_df['win_rate'].mean()
            profit_factor = results_df['profit_factor'].mean()
            
            # Risk-adjusted return
            risk_free_rate = 0.02  # 2% annual risk-free rate
            excess_return = total_return - risk_free_rate
            sortino_ratio = excess_return / results_df['returns'][results_df['returns'] < 0].std() \
                if len(results_df['returns'][results_df['returns'] < 0]) > 0 else 0
                
            return {
                'total_return': float(total_return),
                'sharpe_ratio': float(sharpe_ratio),
                'sortino_ratio': float(sortino_ratio),
                'max_drawdown': float(max_drawdown),
                'win_rate': float(win_rate),
                'profit_factor': float(profit_factor),
                'risk_adjusted_return': float(excess_return / max_drawdown) if max_drawdown != 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing performance: {str(e)}")
            return {}
            
    def analyze_risk_metrics(self, results_df: pd.DataFrame) -> Dict:
        """
        Phân tích các metrics rủi ro
        
        Args:
            results_df: DataFrame chứa kết quả test
            
        Returns:
            Dictionary chứa các metrics rủi ro
        """
        if results_df.empty:
            return {}
            
        try:
            # Value at Risk (VaR)
            returns = results_df['returns'].values
            var_95 = np.percentile(returns, 5)  # 95% VaR
            var_99 = np.percentile(returns, 1)  # 99% VaR
            
            # Expected Shortfall (CVaR)
            cvar_95 = returns[returns <= var_95].mean()
            cvar_99 = returns[returns <= var_99].mean()
            
            # Volatility
            daily_vol = returns.std()
            annualized_vol = daily_vol * np.sqrt(252)  # Assuming 252 trading days
            
            return {
                'var_95': float(var_95),
                'var_99': float(var_99),
                'cvar_95': float(cvar_95) if not np.isnan(cvar_95) else 0,
                'cvar_99': float(cvar_99) if not np.isnan(cvar_99) else 0,
                'daily_volatility': float(daily_vol),
                'annualized_volatility': float(annualized_vol)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing risk metrics: {str(e)}")
            return {}
            
    def plot_performance_metrics(self, results_df: pd.DataFrame, symbol: str, interval: str):
        """
        Vẽ biểu đồ các metrics hiệu suất
        
        Args:
            results_df: DataFrame chứa kết quả test
            symbol: Trading pair
            interval: Timeframe
        """
        if results_df.empty:
            return
            
        try:
            # Tạo figure với 2x2 subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot 1: Returns Distribution
            sns.histplot(results_df['returns'], ax=ax1, kde=True)
            ax1.set_title('Returns Distribution')
            ax1.set_xlabel('Return')
            ax1.set_ylabel('Frequency')
            
            # Plot 2: Drawdown over time
            ax2.plot(results_df.index, results_df['max_drawdown'])
            ax2.set_title('Maximum Drawdown')
            ax2.set_xlabel('Trade')
            ax2.set_ylabel('Drawdown')
            
            # Plot 3: Win Rate vs Profit Factor
            ax3.bar(['Win Rate', 'Profit Factor'], 
                   [results_df['win_rate'].mean(), results_df['profit_factor'].mean()])
            ax3.set_title('Win Rate & Profit Factor')
            
            # Plot 4: Risk Metrics
            risk_metrics = self.analyze_risk_metrics(results_df)
            ax4.bar(['VaR 95%', 'CVaR 95%', 'Daily Vol'],
                   [abs(risk_metrics['var_95']), 
                    abs(risk_metrics['cvar_95']),
                    risk_metrics['daily_volatility']])
            ax4.set_title('Risk Metrics')
            
            plt.tight_layout()
            
            # Lưu plot
            plot_file = os.path.join(
                self.analysis_dir,
                f"performance_analysis_{symbol}_{interval}.png"
            )
            plt.savefig(plot_file)
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting performance metrics: {str(e)}")
            
    def generate_analysis_report(self, symbol: str, interval: str) -> Dict:
        """
        Tạo báo cáo phân tích tổng hợp
        
        Args:
            symbol: Trading pair
            interval: Timeframe
            
        Returns:
            Dictionary chứa báo cáo phân tích
        """
        try:
            # Load test results
            results_df = self.load_test_results(symbol, interval)
            if results_df.empty:
                return {}
                
            # Phân tích hiệu suất
            performance_metrics = self.analyze_trading_performance(results_df)
            risk_metrics = self.analyze_risk_metrics(results_df)
            
            # Tạo plots
            self.plot_performance_metrics(results_df, symbol, interval)
            
            # Tổng hợp báo cáo
            report = {
                'symbol': symbol,
                'interval': interval,
                'performance_metrics': performance_metrics,
                'risk_metrics': risk_metrics,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Lưu báo cáo
            report_file = os.path.join(
                self.analysis_dir,
                f"analysis_report_{symbol}_{interval}.json"
            )
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=4)
                
            return report
            
        except Exception as e:
            logger.error(f"Error generating analysis report: {str(e)}")
            return {}
            
def main():
    """Hàm main để chạy performance analysis"""
    try:
        analyzer = PerformanceAnalyzer()
        
        # Phân tích cho mỗi cặp giao dịch
        for symbol in TRADING_PAIRS:
            for interval in TIMEFRAMES:
                logger.info(f"Analyzing performance for {symbol} {interval}")
                
                report = analyzer.generate_analysis_report(symbol, interval)
                
                if report:
                    logger.info(f"Analysis report for {symbol} {interval}:")
                    logger.info(json.dumps(report, indent=2))
                    
        logger.info("Performance analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error in performance analysis: {str(e)}")

if __name__ == "__main__":
    main() 