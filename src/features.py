"""
features.py

Standard technical indicators computed from OHLCV data. These are the
engineered features that feed both the risk metrics (risk_metrics.py) and
the ML price-direction model (train_predictor.py).

Pure pandas/numpy, no external dependencies beyond what's already in
requirements.txt.
"""

import numpy as np
import pandas as pd


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["daily_return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df


def add_moving_averages(df: pd.DataFrame, windows=(5, 10, 20, 50)) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f"sma_{w}"] = df["close"].rolling(window=w).mean()
        df[f"ema_{w}"] = df["close"].ewm(span=w, adjust=False).mean()
    return df


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    df = df.copy()
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    df = df.copy()
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def add_bollinger_bands(df: pd.DataFrame, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    df = df.copy()
    mid = df["close"].rolling(window=window).mean()
    std = df["close"].rolling(window=window).std()
    df["bb_mid"] = mid
    df["bb_upper"] = mid + n_std * std
    df["bb_lower"] = mid - n_std * std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / mid
    return df


def add_volatility(df: pd.DataFrame, windows=(10, 20, 30)) -> pd.DataFrame:
    df = df.copy()
    if "daily_return" not in df.columns:
        df = add_returns(df)
    for w in windows:
        df[f"volatility_{w}d"] = df["daily_return"].rolling(window=w).std() * np.sqrt(252)  # annualized
    return df


def build_feature_set(df: pd.DataFrame) -> pd.DataFrame:
    """Runs the full feature pipeline in the right order."""
    df = add_returns(df)
    df = add_moving_averages(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_volatility(df)
    return df


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample/IBM_daily.csv"
    df = pd.read_csv(path)
    df = build_feature_set(df)
    print(df.tail(10).to_string())
    print(f"\n{len(df.columns)} columns, {len(df)} rows")
