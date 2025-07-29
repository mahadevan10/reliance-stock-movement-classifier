import pandas as pd
from sklearn.preprocessing import RobustScaler

from data_preprocessing import preprocess_data

# STEP 1: Load, clean, and normalize
stock, volume_scaler, price_scaler = preprocess_data()

# STEP 2: Feature Engineering + Target Creation
def create_features(df):
    df_fe = df.copy()

    # Lags
    df_fe['Close_t-1'] = df_fe['Close'].shift(1)
    df_fe['Volume_t-1'] = df_fe['Volume'].shift(1)

    # Rolling Means & STDs
    df_fe['Close_MA_5'] = df_fe['Close'].rolling(window=5).mean()
    df_fe['Close_MA_10'] = df_fe['Close'].rolling(window=10).mean()
    df_fe['Close_STD_5'] = df_fe['Close'].rolling(window=5).std()

    # Daily Return
    df_fe['Daily_Return'] = (df_fe['Close'] - df_fe['Open']) / df_fe['Open']

    # Volume Change
    df_fe['Volume_Change'] = df_fe['Volume'].pct_change()

    # Optional: Day of Week
    df_fe['Day_of_Week'] = df_fe.index.dayofweek  # 0 = Monday, 4 = Friday

    # Target: 1 if tomorrow's Close > today's, else 0
    df_fe['Target'] = (df_fe['Close'].shift(-1) > df_fe['Close']).astype(int)

    # Drop rows with NaNs (from rolling, shifting)
    df_fe.dropna(inplace=True)

    return df_fe

# STEP 3: Split into Features (X) and Target (y)
def prepare_xy(df_fe):
    feature_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'Close_t-1', 'Volume_t-1',
        'Close_MA_5', 'Close_MA_10', 'Close_STD_5',
        'Daily_Return', 'Volume_Change', 'Day_of_Week'
    ]
    X = df_fe[feature_cols]
    y = df_fe['Target']
    return X, y



# Feature engineering
stock_features = create_features(stock)

# Prepare X and y for training
X, y = prepare_xy(stock_features)

# Inspect final shape
print("Feature shape:", X.shape)
print("Target value counts:\n", y.value_counts())


# STEP 4: Save X and y for model training
import pickle
import os

# Define output path (adjust if needed)
output_dir = r'E:\randomForestClassifier\reliance-stock-rf-classifier\data'
os.makedirs(output_dir, exist_ok=True)

# Save X
with open(os.path.join(output_dir, 'X.pkl'), 'wb') as f:
    pickle.dump(X, f)

# Save y
with open(os.path.join(output_dir, 'y.pkl'), 'wb') as f:
    pickle.dump(y, f)

print("X and y saved successfully to:", output_dir)

