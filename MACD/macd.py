import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Tải dữ liệu cổ phiếu (VD: VNINDEX)
ticker = "BTC-USD"  # Thay mã cổ phiếu bạn muốn
df = yf.download(ticker, period="12mo", interval="1d")  # Lấy dữ liệu 12 tháng

# Tính đường EMA 12 và EMA 26
df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

# Tính MACD và Signal Line
df["MACD"] = df["EMA_12"] - df["EMA_26"]
df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

# Tính Histogram
df["Histogram"] = df["MACD"] - df["Signal_Line"]

# Vẽ biểu đồ
plt.figure(figsize=(12, 6))

# Biểu đồ giá
plt.subplot(2, 1, 1)
plt.plot(df["Close"], label="Giá Đóng Cửa", color="black")
plt.title(f"Biểu đồ giá {ticker}")
plt.legend()

# Biểu đồ MACD
plt.subplot(2, 1, 2)
plt.plot(df["MACD"], label="MACD", color="blue")
plt.plot(df["Signal_Line"], label="Signal Line", color="red")
plt.bar(df.index, df["Histogram"], label="Histogram", color="gray")
plt.axhline(0, color="black", linestyle="--", linewidth=0.5)
plt.title("Chỉ báo MACD")
plt.legend()

plt.tight_layout()
plt.show()
