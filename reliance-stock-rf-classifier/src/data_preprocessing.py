import pandas as pd
import yfinance as yf
from sklearn.preprocessing import RobustScaler

# Load the Excel file
def load_data():
    # Download last 150 days of RELIANCE.NS stock data
    stock = yf.download('RELIANCE.NS', period='150d', interval='1d', progress=False)
    return stock

# Clean the data and round specific columns
def clean_data(stock):
    #stock = stock.iloc[2:]  
    column_names = [ 'Close', 'High', 'Low', 'Open', 'Volume']
    stock.columns = column_names
    columns_to_round = ['Open', 'High', 'Low', 'Close']
    stock[columns_to_round] = stock[columns_to_round].round(2)
    stock.dropna(inplace=True)
    return stock


#normalise data with manual scaling factors
def normalize_data(df):
    df_scaled = df.copy()   
    # Manual scaling factors
    volume_factor = 1_000_000
    price_factor = 1000
    # Apply manual scaling
    df_scaled['Volume'] = df_scaled['Volume'] / volume_factor
    df_scaled[['Open', 'High', 'Low', 'Close']] = df_scaled[['Open', 'High', 'Low', 'Close']] / price_factor
    # Return both the scaled data and scaling factors for rescaling later
    return df_scaled, volume_factor, price_factor

#Preprocess function to call load + clean + normalize
def preprocess_data():
    stock = load_data()
    cleaned_stock = clean_data(stock)
    normalized_stock_df, volume_scaler, price_scaler = normalize_data(cleaned_stock)
    return normalized_stock_df, volume_scaler, price_scaler


# Confirm it worked
stock_df, volume_scaler, price_scaler = preprocess_data()
print("Columns:", stock_df.columns)
print(stock_df.head())
