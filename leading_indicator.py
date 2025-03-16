import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Tải dữ liệu cổ phiếu (VD: BTC-USD hoặc cổ phiếu khác)
ticker = "VNM.VN"  # Thay mã chứng khoán bạn muốn
df = yf.download(ticker, period="12mo", interval="1d")  # Lấy dữ liệu 12 tháng

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

# Vẽ biểu đồ
plt.figure(figsize=(12, 10))

# Biểu đồ giá với Bollinger Bands
plt.subplot(3, 1, 1)
plt.plot(df.index, df["Close"], label="Giá Đóng Cửa", color="black")
plt.plot(df.index, df["SMA_20"], label="SMA 20", color="blue", linestyle="dashed")
plt.plot(df.index, df["Upper_BB"], label="Upper Bollinger Band", color="red", linestyle="dotted")
plt.plot(df.index, df["Lower_BB"], label="Lower Bollinger Band", color="green", linestyle="dotted")
plt.title(f"Biểu đồ Giá với Bollinger Bands ({ticker})")
plt.legend()
plt.grid(True)

# Biểu đồ MACD
plt.subplot(3, 1, 2)
plt.plot(df.index, df["MACD"], label="MACD", color="blue")
plt.plot(df.index, df["Signal_Line"], label="Signal Line", color="red")
plt.bar(df.index, df["Histogram"], label="Histogram", color="gray", alpha=0.5)
plt.axhline(0, color="black", linestyle="--", linewidth=0.5)
plt.title("Chỉ báo MACD")
plt.legend()
plt.grid(True)

# Biểu đồ RSI
plt.subplot(3, 1, 3)
plt.plot(df.index, df["RSI"], label="RSI", color="purple")
plt.axhline(70, color="red", linestyle="--", linewidth=0.5)
plt.axhline(30, color="green", linestyle="--", linewidth=0.5)
plt.title("Chỉ báo RSI")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
