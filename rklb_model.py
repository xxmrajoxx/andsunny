"""
Next-day closing price prediction for RKLB (Rocket Lab) as a public proxy for SpaceX,
which has no public share data since it isn't publicly traded.
"""
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

TICKER = "RKLB"


def load_data(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="3y", auto_adjust=True, progress=False)
    df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    feat["close"] = df["Close"]
    feat["volume"] = df["Volume"]
    feat["return_1d"] = df["Close"].pct_change()
    feat["ma_5"] = df["Close"].rolling(5).mean()
    feat["ma_10"] = df["Close"].rolling(10).mean()
    feat["ma_20"] = df["Close"].rolling(20).mean()
    feat["volatility_5"] = df["Close"].rolling(5).std()
    feat["high_low_spread"] = (df["High"] - df["Low"]) / df["Close"]
    for lag in (1, 2, 3, 5):
        feat[f"close_lag_{lag}"] = df["Close"].shift(lag)

    feat["target_next_close"] = df["Close"].shift(-1)
    return feat.dropna()


def main():
    raw = load_data(TICKER)
    data = build_features(raw)

    feature_cols = [c for c in data.columns if c != "target_next_close"]
    X = data[feature_cols]
    y = data["target_next_close"]

    split = int(len(data) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"Ticker: {TICKER}")
    print(f"Train rows: {len(X_train)}  Test rows: {len(X_test)}")
    print(f"Test MAE:  ${mae:.2f}")
    print(f"Test RMSE: ${rmse:.2f}")

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTop features:")
    print(importances.head(5).to_string())

    latest_features = X.iloc[[-1]]
    next_pred = model.predict(latest_features)[0]
    last_close = data["close"].iloc[-1]
    print(f"\nLast close: ${last_close:.2f}")
    print(f"Predicted next close: ${next_pred:.2f}")


if __name__ == "__main__":
    main()
