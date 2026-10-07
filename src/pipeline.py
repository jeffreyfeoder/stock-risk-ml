"""
pipeline.py

The single entry point: fetch data for a watchlist -> engineer features ->
compute risk metrics -> assign risk categories -> train/evaluate the
next-day direction predictor -> print a summary report.

WHAT "REAL-TIME" MEANS HERE (read this before writing it up): this uses
Alpha Vantage's free-tier daily bars, refreshed whenever you run the
pipeline. That's "as fresh as the free data gets," refreshed on demand or
on a schedule (e.g. once each evening after market close) -- not literal
tick-by-tick live streaming, which requires a paid data feed. Say this
plainly in the report; it's still a legitimate and useful system, just be
accurate about what it is.

Usage:
    # with your free Alpha Vantage key, live data:
    python -m src.pipeline --symbols IBM AAPL MSFT --api_key YOUR_KEY

    # offline, using the bundled real sample (IBM only):
    python -m src.pipeline --offline
"""

import argparse
import os
import pandas as pd

from src.data_fetcher import fetch_watchlist_daily
from src.features import build_feature_set
from src.risk_scoring import score_watchlist
from src.train_predictor import train_and_evaluate


def run_offline(sample_path: str = "data/sample/IBM_daily.csv", symbol: str = "IBM"):
    print("Running in OFFLINE mode on the bundled real sample "
          f"({sample_path}, {symbol} only).\n")
    raw = pd.read_csv(sample_path)
    return {symbol: build_feature_set(raw)}


def run_live(symbols, api_key, out_dir="data/live"):
    print(f"Fetching live daily data for {symbols}...\n")
    raw_dfs = fetch_watchlist_daily(symbols, api_key, out_dir)
    return {sym: build_feature_set(df) for sym, df in raw_dfs.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=["IBM"])
    parser.add_argument("--api_key", type=str, default=os.environ.get("ALPHAVANTAGE_API_KEY"))
    parser.add_argument("--offline", action="store_true", help="use the bundled sample instead of hitting the API")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    if args.offline or not args.api_key:
        if not args.offline:
            print("No API key found (set ALPHAVANTAGE_API_KEY or pass --api_key). "
                  "Falling back to --offline mode.\n")
        feature_dfs = run_offline()
    else:
        feature_dfs = run_live(args.symbols, args.api_key)

    print("=" * 60)
    print("RISK ASSESSMENT SUMMARY")
    print("=" * 60)
    risk_table = score_watchlist(feature_dfs)
    print(risk_table.to_string(index=False))
    risk_table.to_csv(os.path.join(args.results_dir, "risk_summary.csv"), index=False)

    print("\n" + "=" * 60)
    print("NEXT-DAY DIRECTION PREDICTION (per symbol)")
    print("=" * 60)
    for symbol in feature_dfs:
        print(f"\n--- {symbol} ---")
        # train_and_evaluate reloads from CSV for simplicity/consistency;
        # for a live run it re-reads the file data_fetcher.py just saved.
        csv_path = (
            "data/sample/IBM_daily.csv" if (args.offline or not args.api_key)
            else f"data/live/{symbol}_daily.csv"
        )
        try:
            train_and_evaluate(csv_path)
        except Exception as e:
            print(f"  Skipped {symbol}: {e}")

    print(f"\nSaved risk summary to {args.results_dir}/risk_summary.csv")


if __name__ == "__main__":
    main()
