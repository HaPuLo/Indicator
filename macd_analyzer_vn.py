from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import matplotlib.pyplot as plt

API_KEY = "O76P4SPI0P9X1D44"  # Nhập API key tại đây

def fetch_stock_data(symbol):
    """Lấy dữ liệu giá cổ phiếu từ Alpha Vantage"""
    try:
        ts = TimeSeries(key=API_KEY, output_format='pandas')
        df, meta_data = ts.get_daily(symbol=symbol, outputsize='compact')

        if df is None or df.empty:
            print(f"No data found for {symbol}.")
            return None

        df.rename(columns={'4. close': 'Close'}, inplace=True)
        df = df[['Close']].sort_index()  # Chỉ giữ cột giá đóng cửa
        return df
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        return None

def calculate_macd(df):
    """Tính MACD"""
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["Histogram"] = df["MACD"] - df["Signal_Line"]

def plot_stock_chart(df, symbol):
    """Vẽ biểu đồ giá và MACD"""
    plt.figure(figsize=(12, 6))

    # Biểu đồ giá cổ phiếu
    plt.subplot(2, 1, 1)
    plt.plot(df["Close"], label="Giá Đóng Cửa", color="black")
    plt.title(f"Biểu đồ giá {symbol}")
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

def main():
    symbol = 'SSI'  # Mã chứng khoán SSI
    df = fetch_stock_data(symbol)

    if df is not None and not df.empty:
        calculate_macd(df)
        plot_stock_chart(df, symbol)
    else:
        print("Không thể lấy dữ liệu.")

if __name__ == "__main__":
    main()