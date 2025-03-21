import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Tải dữ liệu cổ phiếu (VD: BTC-USD)
ticker = "MWG.VN"
df = yf.download(ticker, period="3mo", interval="1d")  # Dữ liệu 10 ngày, khoảng 1 giờ

# Tính Bollinger Bands (BB)
df["SMA_20"] = df["Close"].rolling(window=20).mean()
df["Std_Dev"] = df["Close"].rolling(window=20).std()
df["Upper_BB"] = df["SMA_20"] + 2 * df["Std_Dev"]
df["Lower_BB"] = df["SMA_20"] - 2 * df["Std_Dev"]

# Tính EMA 12 và EMA 26 cho MACD
df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

# Tính MACD và Signal Line
df["MACD"] = df["EMA_12"] - df["EMA_26"]
df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["Histogram"] = df["MACD"] - df["Signal_Line"]

# Tính RSI
window_length = 14
delta = df["Close"].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=window_length).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=window_length).mean()
rs = gain / loss
df["RSI"] = 100 - (100 / (1 + rs))

# Xử lý NaN
df = df.bfill().dropna()

# Kiểm tra dữ liệu (debug)
print("Dữ liệu DataFrame sau khi xử lý NaN:")
print(df.tail(1))  # In hàng cuối cùng của DataFrame

# Xác định tín hiệu từ các chỉ báo (dựa trên điểm cuối cùng)
latest = df.iloc[-1]  # Lấy dữ liệu mới nhất
print("Dữ liệu latest:")
print(latest)

# Ép buộc lấy giá trị scalar
close_price = float(df["Close"].iloc[-1:].iloc[0])
lower_bb = float(df["Lower_BB"].iloc[-1:].iloc[0])
upper_bb = float(df["Upper_BB"].iloc[-1:].iloc[0])
macd = float(df["MACD"].iloc[-1:].iloc[0])
signal_line = float(df["Signal_Line"].iloc[-1:].iloc[0])
rsi = float(df["RSI"].iloc[-1:].iloc[0])

# Bollinger Bands: bullish nếu giá dưới Lower_BB, bearish nếu trên Upper_BB, trung lập nếu giữa
bb_signal = "not bullish"
if close_price < lower_bb:
    bb_signal = "bullish"
elif close_price > upper_bb:
    bb_signal = "bearish"

# MACD: bullish nếu MACD > Signal Line
macd_signal = "bullish" if macd > signal_line else "not bullish"

# RSI: bullish nếu RSI < 30, bearish nếu RSI > 70, trung lập nếu giữa
rsi_signal = "not bullish"
if rsi < 30:
    rsi_signal = "bullish"
elif rsi > 70:
    rsi_signal = "bearish"

# Tính xác suất giá tăng bằng Bayesian
P_bullish_given_increase = 0.6
P_not_bullish_given_increase = 0.4
P_bullish_given_no_increase = 0.4
P_not_bullish_given_no_increase = 0.6
P_price_increases = 0.5  # Giả định trước

# Tính P(E | price increases) và P(E | price does not increase)
P_E_given_increase = (
    (P_bullish_given_increase if bb_signal == "bullish" else P_not_bullish_given_increase) *
    (P_bullish_given_increase if macd_signal == "bullish" else P_not_bullish_given_increase) *
    (P_bullish_given_increase if rsi_signal == "bullish" else P_not_bullish_given_increase)
)
P_E_given_no_increase = (
    (P_bullish_given_no_increase if bb_signal == "bullish" else P_not_bullish_given_no_increase) *
    (P_bullish_given_no_increase if macd_signal == "bullish" else P_not_bullish_given_no_increase) *
    (P_bullish_given_no_increase if rsi_signal == "bullish" else P_not_bullish_given_no_increase)
)

# Tính P(E)
P_E = P_E_given_increase * P_price_increases + P_E_given_no_increase * (1 - P_price_increases)

# Tính P(price increases | E)
P_price_increases_given_E = (P_E_given_increase * P_price_increases) / P_E

# Dự báo
prediction = "Giá có thể tăng" if P_price_increases_given_E > 0.5 else "Giá có thể giảm hoặc đi ngang"
probability = P_price_increases_given_E * 100

# Vẽ biểu đồ với trục X đồng bộ
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Biểu đồ giá với Bollinger Bands
ax1.plot(df.index, df["Close"], label="Giá Đóng Cửa", color="black")
ax1.plot(df.index, df["SMA_20"], label="SMA 20", color="blue", linestyle="dashed")
ax1.plot(df.index, df["Upper_BB"], label="Upper BB", color="red", linestyle="dotted")
ax1.plot(df.index, df["Lower_BB"], label="Lower BB", color="green", linestyle="dotted")
ax1.set_title(f"Biểu đồ Giá với Bollinger Bands ({ticker})")
ax1.legend()
ax1.grid(True)

# Biểu đồ MACD
ax2.plot(df.index, df["MACD"], label="MACD", color="blue")
ax2.plot(df.index, df["Signal_Line"], label="Signal Line", color="red")
ax2.bar(df.index, df["Histogram"], label="Histogram", color="gray", alpha=0.5)
ax2.axhline(0, color="black", linestyle="--", linewidth=0.5)
ax2.set_title("Chỉ báo MACD")
ax2.legend()
ax2.grid(True)

# Biểu đồ RSI
ax3.plot(df.index, df["RSI"], label="RSI", color="purple")
ax3.axhline(70, color="red", linestyle="--", linewidth=0.5)
ax3.axhline(30, color="green", linestyle="--", linewidth=0.5)
ax3.set_title("Chỉ báo RSI")
ax3.legend()
ax3.grid(True)



# In kết quả chi tiết
print(f"Trạng thái chỉ báo:")
print(f"- Bollinger Bands: {bb_signal}")
print(f"- MACD: {macd_signal}")
print(f"- RSI: {rsi_signal}")
print(f"Dự báo: {prediction}")
print(f"Xác suất giá tăng: {probability:.2f}%")

# Thêm dự báo vào biểu đồ
fig.suptitle(f"Dự báo: {prediction} (Xác suất tăng: {probability:.2f}%)", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()