"""
ticker_lists.py

Fetches the current S&P 500 (US) and Nifty 500 (India) constituent lists
from free, no-auth sources, normalizes each symbol into the exact form
yfinance expects, and caches the combined ~1,000-row universe to
data/universe/tickers.csv.

Sources (fetched fresh each run -- index constituents change a few times a
year, so we don't want a stale hardcoded snapshot):
  - S&P 500: raw.githubusercontent.com/datasets/s-and-p-500-companies
  - Nifty 500: archives.nseindia.com/content/indices/ind_nifty500list.csv

Ticker normalization for yfinance:
  - NSE symbols get a ".NS" suffix, e.g. RELIANCE -> RELIANCE.NS
  - S&P symbols with a "." (share-class tickers like BRK.B) become "-",
    e.g. BRK.B -> BRK-B

Usage:
    python -m src.ticker_lists                # fetch fresh, write cache
    python -m src.ticker_lists --use_cache     # reuse cache if present
"""

import argparse
import os
import pandas as pd
import requests

SP500_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
NIFTY500_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
CACHE_PATH = "data/universe/tickers.csv"

# archives.nseindia.com rejects requests without a browser-like User-Agent
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; risk-screen/1.0)"}


def _sp500_to_yfinance(symbol: str) -> str:
    return symbol.replace(".", "-")


def _nifty_to_yfinance(symbol: str) -> str:
    return f"{symbol.strip()}.NS"


def fetch_sp500() -> pd.DataFrame:
    resp = requests.get(SP500_URL, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(pd.io.common.StringIO(resp.text))
    return pd.DataFrame({
        "ticker": df["Symbol"].map(_sp500_to_yfinance),
        "raw_symbol": df["Symbol"],
        "name": df["Security"],
        "sector": df["GICS Sector"],
        "industry": df["GICS Sub-Industry"],
        "market": "US",
    })


def fetch_nifty500() -> pd.DataFrame:
    resp = requests.get(NIFTY500_URL, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(pd.io.common.StringIO(resp.text))
    return pd.DataFrame({
        "ticker": df["Symbol"].map(_nifty_to_yfinance),
        "raw_symbol": df["Symbol"],
        "name": df["Company Name"],
        "sector": df["Industry"],
        "industry": df["Industry"],
        "market": "India",
    })


def build_universe(use_cache: bool = False, cache_path: str = CACHE_PATH) -> pd.DataFrame:
    if use_cache and os.path.exists(cache_path):
        print(f"Using cached ticker list: {cache_path}")
        return pd.read_csv(cache_path)

    print("Fetching S&P 500 constituents...")
    sp500 = fetch_sp500()
    print(f"  {len(sp500)} US tickers")

    print("Fetching Nifty 500 constituents...")
    nifty500 = fetch_nifty500()
    print(f"  {len(nifty500)} India tickers")

    universe = pd.concat([sp500, nifty500], ignore_index=True)
    universe = universe.drop_duplicates(subset="ticker").reset_index(drop=True)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    universe.to_csv(cache_path, index=False)
    print(f"Saved {len(universe)} tickers to {cache_path}")
    return universe


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_cache", action="store_true",
                         help="reuse data/universe/tickers.csv instead of re-fetching")
    args = parser.parse_args()

    universe = build_universe(use_cache=args.use_cache)
    print(universe["market"].value_counts())
    print(universe.head(10).to_string(index=False))
