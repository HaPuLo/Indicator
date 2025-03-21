import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Tải dữ liệu cổ phiếu (VD: BTC-USD hoặc cổ phiếu khác)
ticker = "BTC-USD"  # Thay mã chứng khoán bạn muốn
df = yf.download(ticker, period="10d", interval="1h")  # Lấy dữ liệu 10 ngày, khoảng 1 giờ

# Tính Bollinger Bands (BB)
df["SMA_20"] = df["Close"].rolling(window=20).mean()  # SMA 20
df["Std_Dev"] = df["Close"].rolling(window=20).std()  # Độ lệch chuẩn
df["Upper_BB"] = df["SMA_20"] + 2 * df["Std_Dev"]  # Dải trên
df["Lower_BB"] = df["SMA_20"] - 2 * df["Std_Dev"]  # Dải dưới

# Tính EMA 12 và EMA 26 cho MACD
df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

# Tính MACD và Signal Line
df["MACD"] = df["EMA_12"] - df["EMA_26"]
df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["Histogram"] = df["MACD"] - df["Signal_Line"]

# Tính RSI (Relative Strength Index)
window_length = 14
delta = df["Close"].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=window_length).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=window_length).mean()
rs = gain / loss
df["RSI"] = 100 - (100 / (1 + rs))

# Vẽ biểu đồ với trục X đồng bộ
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Biểu đồ giá với Bollinger Bands
ax1.plot(df.index, df["Close"], label="Giá Đóng Cửa", color="black")
ax1.plot(df.index, df["SMA_20"], label="SMA 20", color="blue", linestyle="dashed")
ax1.plot(df.index, df["Upper_BB"], label="Upper Bollinger Band", color="red", linestyle="dotted")
ax1.plot(df.index, df["Lower_BB"], label="Lower Bollinger Band", color="green", linestyle="dotted")
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

# Điều chỉnh layout
plt.tight_layout()
plt.show()