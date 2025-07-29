def plot_stock_prices(data):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 7))
    plt.plot(data['Date'], data['Close'], label='Close Price', color='blue')
    plt.title('Reliance Stock Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()

def calculate_moving_average(data, window):
    return data['Close'].rolling(window=window).mean()

def calculate_volatility(data, window):
    return data['Close'].rolling(window=window).std()

def save_to_csv(data, filename):
    data.to_csv(filename, index=False)

def load_from_csv(filename):
    import pandas as pd
    return pd.read_csv(filename)