from vnstock import Vnstock
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

# Khởi tạo đối tượng Stock cho mã VND
stock = Vnstock().stock(symbol='FPT', source='TCBS')

# Lấy dữ liệu lịch sử từ 01/01/2020 đến hiện tại (20/03/2025)
data = stock.quote.history(start='2020-01-01', end='2025-03-20')

# Trích xuất cột giá đóng cửa
close_prices = data['close']

# Chuẩn hóa dữ liệu
scaler = MinMaxScaler(feature_range=(0, 1))
close_prices_scaled = scaler.fit_transform(close_prices.values.reshape(-1, 1))

# Tạo chuỗi dữ liệu cho LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

seq_length = 60  # Sử dụng 60 ngày để dự đoán ngày tiếp theo
X, y = create_sequences(close_prices_scaled, seq_length)

# Chia dữ liệu thành tập huấn luyện và kiểm tra
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape dữ liệu cho LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Xây dựng và huấn luyện mô hình LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Huấn luyện mô hình
model.fit(X_train, y_train, batch_size=1, epochs=10, verbose=1)

# Dự đoán giá đóng cửa ngày tiếp theo
last_sequence = close_prices_scaled[-seq_length:]
last_sequence = last_sequence.reshape((1, seq_length, 1))
predicted_scaled = model.predict(last_sequence)
predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]

print(f"Giá đóng cửa dự đoán cho ngày tiếp theo (21/03/2025): {predicted_price:.2f} VND")