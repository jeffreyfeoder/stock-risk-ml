"""
data_fetcher.py

Pulls stock price data from Alpha Vantage's free API. Free tier gives you
daily bars (and delayed intraday bars) at no cost -- 25 requests/day on
the free key, plenty for a personal watchlist refreshed once or twice a
day.

Get a free key (takes under 20 seconds, no credit card):
    https://www.alphavantage.co/support/#api-key

Then either export it as an environment variable:
    export ALPHAVANTAGE_API_KEY=yourkeyhere
or pass --api_key on the command line.

NOTE: this file was written and validated against a real response fetched
during development (see data/sample/IBM_daily.csv, pulled live from
Alpha Vantage's public demo key) but the live API call itself could not be
executed inside the sandbox this project was built in (outbound network
was restricted to a small allowlist that didn't include Alpha Vantage).
It should work normally from your own machine/Colab -- if it doesn't,
the likely culprit is a response-shape change in Alpha Vantage's API,
which is easy to patch in `_parse_time_series`.
"""

import argparse
import os
import time
import requests
import pandas as pd

BASE_URL = "https://www.alphavantage.co/query"


def _parse_time_series(raw_json: dict, series_key: str) -> pd.DataFrame:
    if series_key not in raw_json:
        raise ValueError(f"Unexpected API response: {raw_json}")
    series = raw_json[series_key]
    rows = []
    for date, vals in series.items():
        rows.append({
            "date": date,
            "open": float(vals["1. open"]),
            "high": float(vals["2. high"]),
            "low": float(vals["3. low"]),
            "close": float(vals["4. close"]),
            "volume": int(float(vals["5. volume"])),
        })
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


def fetch_daily(symbol: str, api_key: str, outputsize: str = "compact") -> pd.DataFrame:
    """outputsize: 'compact' = last 100 days, 'full' = 20+ years."""
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": outputsize,
        "apikey": api_key,
    }
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    return _parse_time_series(resp.json(), "Time Series (Daily)")


def fetch_intraday(symbol: str, api_key: str, interval: str = "5min") -> pd.DataFrame:
    """interval: '1min', '5min', '15min', '30min', '60min'. Free tier data
    is delayed, not tick-level -- this is the "as real-time as the free
    tier gets" option, be upfront about that in any write-up."""
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": interval,
        "apikey": api_key,
    }
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    return _parse_time_series(resp.json(), f"Time Series ({interval})")


def fetch_news_sentiment(symbol: str, api_key: str, limit: int = 50) -> dict:
    """Alpha Vantage's NEWS_SENTIMENT endpoint: recent news articles for a
    ticker with Alpha Vantage's own per-article sentiment score attached
    (both an overall score and a ticker-specific one). Same free API key as
    the price endpoints above.

    NOTE: unlike TIME_SERIES_DAILY, the free-tier 'demo' key does not
    return real data for NEWS_SENTIMENT for any symbol -- there's no
    demo-key sample to validate against here. This function's shape is
    correct per Alpha Vantage's documented response, and src/sentiment.py's
    aggregation logic is validated against a hand-built realistic sample
    (data/sample/AAPL_news_sentiment_sample.json) instead. It should work
    normally with your own free key -- see
    https://www.alphavantage.co/support/#api-key.
    """
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": symbol,
        "limit": str(limit),
        "apikey": api_key,
    }
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "feed" not in data:
        raise ValueError(f"Unexpected NEWS_SENTIMENT response for {symbol}: {data}")
    return data


def fetch_watchlist_daily(symbols: list, api_key: str, out_dir: str = "data/live"):
    """Fetches daily data for each symbol, respecting the free tier's rate
    limit (5 requests/minute) by pausing between calls."""
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    for i, symbol in enumerate(symbols):
        print(f"Fetching {symbol} ({i+1}/{len(symbols)})...")
        df = fetch_daily(symbol, api_key)
        df.to_csv(os.path.join(out_dir, f"{symbol}_daily.csv"), index=False)
        results[symbol] = df
        if i < len(symbols) - 1:
            time.sleep(13)  # stay under 5 req/min on the free tier
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=["IBM"])
    parser.add_argument("--api_key", type=str, default=os.environ.get("ALPHAVANTAGE_API_KEY", "demo"))
    parser.add_argument("--out_dir", type=str, default="data/live")
    args = parser.parse_args()

    if args.api_key == "demo" and args.symbols != ["IBM"]:
        print("WARNING: the 'demo' API key only returns real data for IBM. "
              "Get a free key at https://www.alphavantage.co/support/#api-key "
              "to fetch other symbols.")

    fetch_watchlist_daily(args.symbols, args.api_key, args.out_dir)
    print(f"Saved to {args.out_dir}/")
