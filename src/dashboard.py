import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from binance.client import Client
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRADING_PAIRS, TIMEFRAMES, PATHS
from performance_analysis import PerformanceAnalyzer

# Thiết lập page config
st.set_page_config(
    page_title="Trading Bot Dashboard",
    page_icon="📈",
    layout="wide"
)

# Styles
st.markdown("""
    <style>
    .big-font {
        font-size:24px !important;
        font-weight: bold;
    }
    .medium-font {
        font-size:18px !important;
    }
    .small-font {
        font-size:14px !important;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

class Dashboard:
    def __init__(self):
        """Khởi tạo Dashboard"""
        from dotenv import load_dotenv
        load_dotenv()
        
        # Kết nối Binance API
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_SECRET_KEY')
        self.client = Client(self.api_key, self.api_secret)
        
        # Khởi tạo Performance Analyzer
        self.analyzer = PerformanceAnalyzer()
        
    def load_account_info(self) -> dict:
        """Load thông tin tài khoản"""
        try:
            account = self.client.get_account()
            balances = {}
            
            for asset in account['balances']:
                free = float(asset['free'])
                locked = float(asset['locked'])
                if free > 0 or locked > 0:
                    balances[asset['asset']] = {
                        'free': free,
                        'locked': locked,
                        'total': free + locked
                    }
                    
            return {
                'balances': balances,
                'can_trade': account['canTrade'],
                'can_withdraw': account['canWithdraw'],
                'update_time': datetime.fromtimestamp(account['updateTime']/1000)
            }
            
        except Exception as e:
            st.error(f"Error loading account info: {str(e)}")
            return {}
            
    def load_open_orders(self) -> pd.DataFrame:
        """Load các lệnh đang mở"""
        try:
            orders = []
            for symbol in TRADING_PAIRS:
                open_orders = self.client.get_open_orders(symbol=symbol)
                orders.extend(open_orders)
                
            if not orders:
                return pd.DataFrame()
                
            df = pd.DataFrame(orders)
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            df['price'] = df['price'].astype(float)
            df['origQty'] = df['origQty'].astype(float)
            df['executedQty'] = df['executedQty'].astype(float)
            
            return df
            
        except Exception as e:
            st.error(f"Error loading open orders: {str(e)}")
            return pd.DataFrame()
            
    def load_trade_history(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Load lịch sử giao dịch"""
        try:
            trades = self.client.get_my_trades(symbol=symbol, limit=limit)
            
            if not trades:
                return pd.DataFrame()
                
            df = pd.DataFrame(trades)
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            df['price'] = df['price'].astype(float)
            df['qty'] = df['qty'].astype(float)
            df['quoteQty'] = df['quoteQty'].astype(float)
            
            return df
            
        except Exception as e:
            st.error(f"Error loading trade history: {str(e)}")
            return pd.DataFrame()
            
    def load_performance_metrics(self, symbol: str, interval: str) -> dict:
        """Load metrics hiệu suất"""
        try:
            report = self.analyzer.generate_analysis_report(symbol, interval)
            return report.get('performance_metrics', {})
            
        except Exception as e:
            st.error(f"Error loading performance metrics: {str(e)}")
            return {}
            
    def plot_price_chart(self, symbol: str, interval: str):
        """Vẽ biểu đồ giá"""
        try:
            # Lấy dữ liệu giá
            klines = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=100
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignored'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Tạo candlestick chart
            fig = go.Figure(data=[go.Candlestick(
                x=df['timestamp'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close']
            )])
            
            fig.update_layout(
                title=f"{symbol} Price Chart ({interval})",
                yaxis_title="Price",
                xaxis_title="Time"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error plotting price chart: {str(e)}")
            
    def render_dashboard(self):
        """Render toàn bộ dashboard"""
        st.title("Trading Bot Dashboard")
        
        # Sidebar
        st.sidebar.header("Settings")
        selected_pair = st.sidebar.selectbox("Select Trading Pair", TRADING_PAIRS)
        selected_timeframe = st.sidebar.selectbox("Select Timeframe", TIMEFRAMES)
        
        # Main content
        col1, col2, col3 = st.columns(3)
        
        # Account Overview
        with col1:
            st.markdown('<p class="big-font">Account Overview</p>', unsafe_allow_html=True)
            account_info = self.load_account_info()
            
            if account_info:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                for asset, balance in account_info['balances'].items():
                    st.markdown(f"""
                        <p class="medium-font">{asset}</p>
                        <p class="small-font">Free: {balance['free']:.8f}</p>
                        <p class="small-font">Locked: {balance['locked']:.8f}</p>
                        <hr>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
        # Performance Metrics
        with col2:
            st.markdown('<p class="big-font">Performance Metrics</p>', unsafe_allow_html=True)
            metrics = self.load_performance_metrics(selected_pair, selected_timeframe)
            
            if metrics:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f"""
                    <p class="medium-font">Total Return: {metrics['total_return']:.2%}</p>
                    <p class="small-font">Win Rate: {metrics['win_rate']:.2%}</p>
                    <p class="small-font">Sharpe Ratio: {metrics['sharpe_ratio']:.2f}</p>
                    <p class="small-font">Max Drawdown: {metrics['max_drawdown']:.2%}</p>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
        # Open Orders
        with col3:
            st.markdown('<p class="big-font">Open Orders</p>', unsafe_allow_html=True)
            orders_df = self.load_open_orders()
            
            if not orders_df.empty:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                for _, order in orders_df.iterrows():
                    st.markdown(f"""
                        <p class="medium-font">{order['symbol']}</p>
                        <p class="small-font">Type: {order['type']}</p>
                        <p class="small-font">Price: {float(order['price']):.8f}</p>
                        <p class="small-font">Quantity: {float(order['origQty']):.8f}</p>
                        <hr>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No open orders")
                
        # Price Chart
        st.markdown('<p class="big-font">Price Chart</p>', unsafe_allow_html=True)
        self.plot_price_chart(selected_pair, selected_timeframe)
        
        # Trade History
        st.markdown('<p class="big-font">Trade History</p>', unsafe_allow_html=True)
        trades_df = self.load_trade_history(selected_pair)
        
        if not trades_df.empty:
            st.dataframe(
                trades_df[['time', 'price', 'qty', 'quoteQty', 'isBuyer']],
                use_container_width=True
            )
        else:
            st.info("No trade history")
            
def main():
    """Hàm main để chạy dashboard"""
    try:
        dashboard = Dashboard()
        dashboard.render_dashboard()
        
    except Exception as e:
        st.error(f"Error in dashboard: {str(e)}")

if __name__ == "__main__":
    main() 