import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def calculate_RSI(data, periods):
    delta = data.diff()
    up = delta.where(delta > 0, 0)
    down = -delta.where(delta < 0, 0)
    sma_up = up.rolling(window=periods).mean()
    sma_down = down.rolling(window=periods).mean()
    rsi = 100.0 - (100.0 / (1.0 + sma_up / sma_down))
    return rsi

def calculate_MACD(data):
    ema_12 = data.ewm(span=12, adjust=False).mean()
    ema_26 = data.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram

def predict_price_change(model, preprocessed_data):
    pred = model.predict(preprocessed_data)
    if pred == 1:
        return "Buy"
    else:
        return "Sell"

# download stock data
symbol = input("Enter stock symbol: ")
start_date = '2010-01-01'
end_date = '2022-03-11'
df = yf.download(symbol, start=start_date, end=end_date)

# preprocess data
df['SMA_20'] = df['Adj Close'].rolling(window=20).mean()
df['SMA_50'] = df['Adj Close'].rolling(window=50).mean()
df['RSI'] = calculate_RSI(df['Adj Close'], 14)
df['MACD'], df['Signal'], df['Histogram'] = calculate_MACD(df['Adj Close'])
df['Price_Change'] = np.where(df['Adj Close'].shift(-1) > df['Adj Close'], 1, 0)
df.dropna(inplace=True)

# feature engineering
X = df[['SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal', 'Histogram']]
y = df['Price_Change']

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# preprocess data
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal', 'Histogram'])
])

# train model
svm = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', SVC(kernel='linear'))
])

svm.fit(X_train, y_train)

# evaluate model
y_pred = svm.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2f}")

# print classification report and confusion matrix
print(classification_report(y_test, y_pred, zero_division=0))
print(confusion_matrix(y_test, y_pred))

# make predictions on new data
last_row = X.tail(1)
last_row_scaled = preprocessor.transform(last_row)
last_row_scaled_df = pd.DataFrame(last_row_scaled, columns=X.columns.tolist())
pred = predict_price_change(svm, last_row_scaled_df)
print(f"Predicted action: {pred}")

# generate buy/sell signal
if pred == 'Buy':
    print("Buy signal")
else:
    print("Sell signal")