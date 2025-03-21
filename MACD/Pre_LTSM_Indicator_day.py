from vnstock import Vnstock
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from ta.trend import MACD
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Biến global để lưu thông số
MACD_WINDOW_FAST = None
MACD_WINDOW_SLOW = None
MACD_WINDOW_SIGN = None
BB_WINDOW = None
BB_WINDOW_DEV = None
RSI_WINDOW = None
RESAMPLE_PERIOD = None
SEQ_LENGTH = None

def init(strategy="hodl"):
    """Khởi tạo thông số dựa trên chiến lược giao dịch"""
    global MACD_WINDOW_FAST, MACD_WINDOW_SLOW, MACD_WINDOW_SIGN, BB_WINDOW, BB_WINDOW_DEV, RSI_WINDOW, RESAMPLE_PERIOD, SEQ_LENGTH
    
    if strategy.lower() == "scalping":
        MACD_WINDOW_FAST, MACD_WINDOW_SLOW, MACD_WINDOW_SIGN = 6, 13, 5
        BB_WINDOW, BB_WINDOW_DEV = 10, 2
        RSI_WINDOW = 7
        RESAMPLE_PERIOD = '1d'
        SEQ_LENGTH = 5
    elif strategy.lower() == "day_trading":
        MACD_WINDOW_FAST, MACD_WINDOW_SLOW, MACD_WINDOW_SIGN = 9, 21, 5
        BB_WINDOW, BB_WINDOW_DEV = 15, 2
        RSI_WINDOW = 10
        RESAMPLE_PERIOD = '1d'
        SEQ_LENGTH = 8
    elif strategy.lower() == "swing_trading":
        MACD_WINDOW_FAST, MACD_WINDOW_SLOW, MACD_WINDOW_SIGN = 12, 26, 9
        BB_WINDOW, BB_WINDOW_DEV = 15, 2
        RSI_WINDOW = 10
        RESAMPLE_PERIOD = '1d'
        SEQ_LENGTH = 10
    elif strategy.lower() == "hodl":
        MACD_WINDOW_FAST, MACD_WINDOW_SLOW, MACD_WINDOW_SIGN = 24, 52, 18
        BB_WINDOW, BB_WINDOW_DEV = 50, 2.5
        RSI_WINDOW = 21
        RESAMPLE_PERIOD = '1w'
        SEQ_LENGTH = 10
    else:
        raise ValueError("Chiến lược không hợp lệ! Chọn: 'scalping', 'day_trading', 'swing_trading', 'hodl'")

def run_stock_prediction(stock_tick="FPT", strategy="hodl"):
    """Chạy dự đoán giá cổ phiếu với mã và chiến lược cho trước"""
    # Khởi tạo thông số
    init(strategy)
    
    # Khởi tạo đối tượng Stock
    stock = Vnstock().stock(symbol=stock_tick, source='TCBS')
    
    # Lấy dữ liệu lịch sử từ 01/01/2020 đến hiện tại (20/03/2025)
    data = stock.quote.history(start='2020-01-01', end='2025-03-20')
    
    # Kiểm tra dữ liệu gốc
    if data.empty:
        raise ValueError(f"Không có dữ liệu cho cổ phiếu {stock_tick} từ nguồn TCBS!")
    print(f"Số dòng dữ liệu gốc: {len(data)}")
    
    # Chuyển đổi dữ liệu theo khung thời gian
    data['time'] = pd.to_datetime(data['time'])
    data.set_index('time', inplace=True)
    resampled_data = data.resample(RESAMPLE_PERIOD).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    print(f"Số dòng sau resampling ({RESAMPLE_PERIOD}): {len(resampled_data)}")
    
    # Tính toán các chỉ báo kỹ thuật
    macd = MACD(close=resampled_data['close'], window_slow=MACD_WINDOW_SLOW, window_fast=MACD_WINDOW_FAST, window_sign=MACD_WINDOW_SIGN)
    resampled_data['macd'] = macd.macd()
    resampled_data['macd_signal'] = macd.macd_signal()
    
    bb = BollingerBands(close=resampled_data['close'], window=BB_WINDOW, window_dev=BB_WINDOW_DEV)
    resampled_data['bb_upper'] = bb.bollinger_hband()
    resampled_data['bb_lower'] = bb.bollinger_lband()
    resampled_data['bb_mid'] = bb.bollinger_mavg()
    
    rsi = RSIIndicator(close=resampled_data['close'], window=RSI_WINDOW)
    resampled_data['rsi'] = rsi.rsi()
    
    # Điền giá trị NaN
    resampled_data = resampled_data.ffill().bfill()  # Sửa deprecated method
    print(f"Số dòng sau khi điền NaN: {len(resampled_data)}")
    
    # Kiểm tra NaN còn sót lại
    if resampled_data.isna().any().any():
        print("Cảnh báo: Dữ liệu vẫn chứa NaN sau khi điền, điền thêm bằng 0!")
        resampled_data = resampled_data.fillna(0)
    
    # Kiểm tra dữ liệu sau khi xử lý
    if resampled_data.empty:
        raise ValueError(f"Dữ liệu sau khi tính chỉ báo rỗng cho {stock_tick} với khung {RESAMPLE_PERIOD}!")
    
    # Chọn đặc trưng
    features = ['close', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'bb_mid', 'rsi', 'volume']
    data_features = resampled_data[features]
    
    # Chuẩn hóa dữ liệu cho LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_features)
    
    # Tạo chuỗi dữ liệu cho LSTM
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(seq_length, len(data)):
            X.append(data[i-seq_length:i])
            y.append(data[i, 0])  # Dự đoán giá đóng cửa
        return np.array(X), np.array(y)
    
    X, y = create_sequences(scaled_data, SEQ_LENGTH)
    
    # Kiểm tra dữ liệu chuỗi
    if len(X) == 0:
        raise ValueError(f"Không đủ dữ liệu để tạo chuỗi với SEQ_LENGTH={SEQ_LENGTH} cho khung {RESAMPLE_PERIOD} (số mẫu: {len(scaled_data)})!")
    print(f"Số mẫu chuỗi LSTM: {len(X)}")
    
    # Chia dữ liệu
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Xây dựng mô hình LSTM
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, batch_size=1, epochs=10, callbacks=[early_stopping], verbose=1)
    
    # Đánh giá mô hình
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(np.hstack((y_pred_scaled, X_test[:, -1, 1:])))[:, 0]
    y_test_real = scaler.inverse_transform(np.hstack((y_test.reshape(-1, 1), X_test[:, -1, 1:])))[:, 0]
    
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred))
    mae = mean_absolute_error(y_test_real, y_pred)
    print(f"RMSE trên tập kiểm tra ({RESAMPLE_PERIOD}): {rmse:.2f} VND")
    print(f"MAE trên tập kiểm tra ({RESAMPLE_PERIOD}): {mae:.2f} VND")
    
    # Dự đoán giá
    last_sequence = scaled_data[-SEQ_LENGTH:]
    last_sequence = last_sequence.reshape((1, SEQ_LENGTH, len(features)))
    predicted_scaled = model.predict(last_sequence)
    predicted_scaled_full = np.zeros((1, len(features)))
    predicted_scaled_full[:, 0] = predicted_scaled
    predicted_scaled_full[:, 1:] = scaled_data[-1, 1:]
    predicted_price = scaler.inverse_transform(predicted_scaled_full)[0, 0]
    
    print(f"Giá đóng cửa dự đoán cho kỳ tiếp theo ({RESAMPLE_PERIOD} từ 23/03/2025): {predicted_price:.2f} VND")
    
    # Mô hình Bayes
    resampled_data['trend'] = 0
    resampled_data['trend'] = np.where(resampled_data['close'].shift(-1) > resampled_data['close'] * 1.02, 1, resampled_data['trend'])
    resampled_data['trend'] = np.where(resampled_data['close'].shift(-1) < resampled_data['close'] * 0.98, -1, resampled_data['trend'])
    
    bayes_data = resampled_data[:-1]
    X_bayes = bayes_data[features]
    y_bayes = bayes_data['trend']
    
    # Kiểm tra NaN trong X_bayes
    if X_bayes.isna().any().any():
        print("Cảnh báo: X_bayes chứa NaN, điền bằng 0!")
        X_bayes = X_bayes.fillna(0)
    
    bayes_scaler = MinMaxScaler(feature_range=(0, 1))
    X_bayes_scaled = bayes_scaler.fit_transform(X_bayes)
    X_train_bayes, X_test_bayes, y_train_bayes, y_test_bayes = train_test_split(X_bayes_scaled, y_bayes, test_size=0.2, random_state=42)
    
    gnb = GaussianNB()
    gnb.fit(X_train_bayes, y_train_bayes)
    bayes_accuracy = gnb.score(X_test_bayes, y_test_bayes)
    print(f"Độ chính xác của mô hình Bayes trên tập kiểm tra ({RESAMPLE_PERIOD}): {bayes_accuracy:.2%}")
    
    last_features = resampled_data[features].iloc[-1:]
    if last_features.isna().any().any():
        print("Cảnh báo: last_features chứa NaN, điền bằng 0!")
        last_features = last_features.fillna(0)
    last_features_scaled = bayes_scaler.transform(last_features)
    trend_prob = gnb.predict_proba(last_features_scaled)[0]
    trend_pred = gnb.predict(last_features_scaled)[0]
    
    trend_labels = {-1: 'Giảm', 0: 'Đi ngang', 1: 'Tăng'}
    print(f"Nhận định xu hướng cho kỳ tiếp theo ({RESAMPLE_PERIOD} từ 23/03/2025): {trend_labels[trend_pred]}")
    print(f"Xác suất (Bayes): Tăng: {trend_prob[2]:.2%}, Đi ngang: {trend_prob[1]:.2%}, Giảm: {trend_prob[0]:.2%}")
    
    # Vẽ biểu đồ
    dates = resampled_data.index
    actual_prices = resampled_data['close']
    test_start_idx = train_size
    pred_dates = dates[test_start_idx:test_start_idx + len(y_pred)]
    pred_dates = list(pred_dates) + [pd.Timestamp('2025-03-23')]
    pred_prices = list(y_pred) + [predicted_price]
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    
    ax1.plot(dates, actual_prices, label=f'Giá thực tế ({RESAMPLE_PERIOD})', color='blue')
    ax1.plot(pred_dates, pred_prices, label=f'Giá dự đoán ({RESAMPLE_PERIOD})', color='red', linestyle='--')
    ax1.set_title(f'Giá cổ phiếu {stock_tick} - {strategy.capitalize()} ({RESAMPLE_PERIOD})')
    ax1.set_ylabel('Giá (VND)')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(dates, resampled_data['close'], label=f'Giá đóng cửa ({RESAMPLE_PERIOD})', color='blue', alpha=0.5)
    ax2.plot(dates, resampled_data['bb_upper'], label='BB Upper', color='green')
    ax2.plot(dates, resampled_data['bb_lower'], label='BB Lower', color='red')
    ax2.plot(dates, resampled_data['bb_mid'], label='BB Mid', color='orange', linestyle='--')
    ax2.set_title(f'Bollinger Bands ({RESAMPLE_PERIOD}) - Window: {BB_WINDOW}, Dev: {BB_WINDOW_DEV}')
    ax2.set_ylabel('Giá (VND)')
    ax2.legend()
    ax2.grid(True)
    
    ax3.plot(dates, resampled_data['macd'], label='MACD', color='blue')
    ax3.plot(dates, resampled_data['macd_signal'], label='Signal', color='orange')
    ax3.set_title(f'MACD ({RESAMPLE_PERIOD}) - {MACD_WINDOW_FAST}, {MACD_WINDOW_SLOW}, {MACD_WINDOW_SIGN}')
    ax3.set_ylabel('Giá trị')
    ax3.legend()
    ax3.grid(True)
    
    ax4.plot(dates, resampled_data['rsi'], label=f'RSI ({RSI_WINDOW})', color='purple')
    ax4.axhline(70, linestyle='--', color='red', alpha=0.5)
    ax4.axhline(40 if strategy.lower() == "hodl" else 30, linestyle='--', color='green', alpha=0.5)
    ax4.set_title(f'RSI ({RESAMPLE_PERIOD}) - Window: {RSI_WINDOW}')
    ax4.set_xlabel('Thời gian')
    ax4.set_ylabel('Giá trị')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()

# Chạy chương trình với mã cổ phiếu và chiến lược mong muốn
if __name__ == "__main__":
    stock_symbol = "TPB"  # Có thể thay đổi: "VND", "PNJ", v.v.
    trading_strategy = "day_trading"  # Chọn: "scalping", "day_trading", "swing_trading", "hodl"
    try:
        run_stock_prediction(stock_symbol, trading_strategy)
    except ValueError as e:
        print(f"Lỗi: {e}")