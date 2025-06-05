# Trading Bot Dashboard

Giao diện web đơn giản để theo dõi bot trading và tài khoản Binance.

## Tính năng

- Theo dõi số dư tài khoản
- Xem các lệnh đang mở
- Xem lịch sử giao dịch
- Biểu đồ giá realtime
- Phân tích hiệu suất
- Metrics về rủi ro

## Cài đặt

1. Clone repository:
```bash
git clone <repository_url>
cd <repository_name>
```

2. Tạo và kích hoạt môi trường ảo:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Cài đặt các thư viện:
```bash
pip install -r requirements.txt
```

4. Tạo file `.env` và thêm API keys:
```
BINANCE_API_KEY=your_api_key
BINANCE_SECRET_KEY=your_secret_key
```

## Sử dụng

1. Chạy dashboard:
```bash
cd src
streamlit run dashboard.py
```

2. Mở trình duyệt và truy cập:
```
http://localhost:8501
```

## Cấu trúc thư mục

```
├── src/
│   ├── dashboard.py        # Giao diện web
│   ├── testing.py          # Backtesting engine
│   ├── optimization.py     # Model optimization
│   ├── performance_analysis.py  # Performance metrics
│   └── run_testing.py     # Testing pipeline
├── .env                   # API keys
└── requirements.txt       # Dependencies
```

## Hướng dẫn sử dụng

1. **Account Overview**: Hiển thị số dư của các coin trong tài khoản

2. **Performance Metrics**: Các chỉ số hiệu suất:
   - Total Return: Lợi nhuận tổng
   - Win Rate: Tỷ lệ thắng
   - Sharpe Ratio: Chỉ số Sharpe
   - Max Drawdown: Mức sụt giảm tối đa

3. **Open Orders**: Danh sách các lệnh đang mở:
   - Symbol: Cặp giao dịch
   - Type: Loại lệnh
   - Price: Giá
   - Quantity: Khối lượng

4. **Price Chart**: Biểu đồ nến theo thời gian thực

5. **Trade History**: Lịch sử các giao dịch đã thực hiện

## Lưu ý

- Đảm bảo API keys có quyền đọc thông tin tài khoản
- Không chia sẻ API keys với người khác
- Nên sử dụng API keys chỉ có quyền đọc để an toàn

## Contributing

Mọi đóng góp đều được hoan nghênh. Vui lòng tạo issue hoặc pull request. 



# Bước 1: Setup môi trường
python main.py 1 --force-setup

# Bước 2: Thu thập dữ liệu cho BTCUSDT và ETHUSDT
python main.py 2 --symbols BTCUSDT ETHUSDT --days 60

# Bước 3: Xây dựng model
python main.py 3 --symbols BTCUSDT ETHUSDT

# Bước 4: Training với 200 epochs
python main.py 4 --symbols BTCUSDT ETHUSDT --epochs 200

# Bước 5: Testing và tối ưu
python main.py 5 --symbols BTCUSDT ETHUSDT

# Bước 6: Khởi động bot với 3 cặp giao dịch
python main.py 6 --symbols BTCUSDT ETHUSDT BNBUSDT