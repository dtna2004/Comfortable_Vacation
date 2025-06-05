import os
import sys
import logging
from datetime import datetime, timedelta
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRADING_PAIRS, TIMEFRAMES, PATHS
from testing import TradingTester
from optimization import ModelOptimizer
from performance_analysis import PerformanceAnalyzer

# Thiết lập logging
os.makedirs(PATHS['logs_dir'], exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PATHS['logs_dir'], 'run_testing.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_testing_pipeline():
    """Chạy toàn bộ quy trình testing và tối ưu"""
    try:
        start_time = datetime.now()
        logger.info("Starting testing pipeline")
        
        # Khởi tạo các components
        tester = TradingTester()
        optimizer = ModelOptimizer()
        analyzer = PerformanceAnalyzer()
        
        # Tạo thư mục cho final results
        final_results_dir = os.path.join(PATHS['data_dir'], 'final_results')
        os.makedirs(final_results_dir, exist_ok=True)
        
        final_results = {}
        
        # Chạy pipeline cho mỗi cặp giao dịch
        for symbol in TRADING_PAIRS:
            final_results[symbol] = {}
            
            for interval in TIMEFRAMES:
                logger.info(f"Processing {symbol} {interval}")
                
                try:
                    # 1. Tối ưu hyperparameters
                    logger.info("Optimizing model hyperparameters...")
                    best_params = optimizer.optimize_hyperparameters(symbol, interval)
                    
                    if not best_params:
                        logger.warning(f"Skipping {symbol} {interval} due to optimization failure")
                        continue
                        
                    # 2. Chạy backtest với params tối ưu
                    logger.info("Running backtest with optimized parameters...")
                    backtest_results = tester.run_backtest(
                        symbol=symbol,
                        interval=interval,
                        start_date=datetime.now() - timedelta(days=30),
                        risk_params=best_params
                    )
                    
                    if not backtest_results:
                        logger.warning(f"Skipping {symbol} {interval} due to backtest failure")
                        continue
                        
                    # 3. Phân tích hiệu suất
                    logger.info("Analyzing performance...")
                    analysis_report = analyzer.generate_analysis_report(symbol, interval)
                    
                    if not analysis_report:
                        logger.warning(f"Skipping {symbol} {interval} due to analysis failure")
                        continue
                        
                    # 4. Tổng hợp kết quả
                    final_results[symbol][interval] = {
                        'best_params': best_params,
                        'backtest_results': backtest_results,
                        'analysis_report': analysis_report
                    }
                    
                    logger.info(f"Successfully processed {symbol} {interval}")
                    
                except Exception as e:
                    logger.error(f"Error processing {symbol} {interval}: {str(e)}")
                    continue
                    
        # Lưu kết quả cuối cùng
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_results_file = os.path.join(
            final_results_dir,
            f"final_results_{timestamp}.json"
        )
        
        with open(final_results_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'execution_time': str(datetime.now() - start_time),
                'results': final_results
            }, f, indent=4)
            
        logger.info(f"Testing pipeline completed in {datetime.now() - start_time}")
        logger.info(f"Results saved to {final_results_file}")
        
        return final_results
        
    except Exception as e:
        logger.error(f"Error in testing pipeline: {str(e)}")
        return {}

def main():
    """Hàm main để chạy testing pipeline"""
    try:
        results = run_testing_pipeline()
        
        if results:
            # In tổng quan kết quả
            logger.info("\nTesting Results Summary:")
            for symbol in results:
                for interval in results[symbol]:
                    res = results[symbol][interval]
                    logger.info(f"\n{symbol} {interval}:")
                    logger.info(f"Best Parameters: {json.dumps(res['best_params'], indent=2)}")
                    logger.info(f"Performance Metrics:")
                    logger.info(json.dumps(res['analysis_report']['performance_metrics'], indent=2))
                    
        logger.info("Testing completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main() 