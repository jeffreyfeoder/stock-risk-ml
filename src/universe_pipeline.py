"""
universe_pipeline.py

Top-level entry point for the full-universe (S&P 500 + Nifty 500) scan:

    ticker_lists -> bulk_fetcher -> risk_screen -> train_pooled_predictor
    -> combined report (risk category + predicted direction + confidence,
       sorted by risk score)

This is the ~1,000-stock counterpart to pipeline.py, which stays as the
small-watchlist / Alpha Vantage entry point and is unaffected by any of
this -- both coexist, use whichever fits the task.

Runtime, honestly: fetching ~1,000 tickers via yfinance takes real time
(rate-limited batches with a delay between them) -- expect roughly 15-30
minutes for a full cold-cache run, and seconds to a couple of minutes on
a re-run once data/universe/*.csv is populated (bulk_fetcher skips
already-cached tickers by default). The risk screen and pooled model
training are fast in comparison (a minute or two) since they're vectorized
computation and one model fit, not 1,000 model fits.

Usage:
    # smoke test on 20 tickers end-to-end
    python -m src.universe_pipeline --limit 20

    # full universe, using whatever's already cached (fast if warm)
    python -m src.universe_pipeline

    # full universe, force re-fetch every ticker's price history
    python -m src.universe_pipeline --refresh
"""

import argparse
import os
import pandas as pd

from src.ticker_lists import build_universe
from src.bulk_fetcher import bulk_fetch
from src.risk_screen import screen_universe
from src.train_pooled_predictor import build_pooled_dataset, train_and_evaluate_pooled
from src.export_stock_detail import export_all as export_stock_detail_all

COMBINED_REPORT_PATH = "results/universe_combined_report.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only use the first N universe tickers (smoke testing)")
    parser.add_argument("--period", type=str, default="2y", help="yfinance history window")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--refresh", action="store_true", help="force re-fetch all price history")
    parser.add_argument("--use_ticker_cache", action="store_true", default=True,
                         help="reuse data/universe/tickers.csv instead of re-fetching index constituents (default on)")
    parser.add_argument("--refresh_tickers", action="store_true", help="re-fetch index constituent lists (S&P 500 / Nifty 500) fresh")
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    print("=" * 70)
    print("STEP 1/5: Ticker universe (S&P 500 + Nifty 500)")
    print("=" * 70)
    universe = build_universe(use_cache=not args.refresh_tickers)
    if args.limit:
        universe = universe.head(args.limit)
    print(f"Universe size: {len(universe)} tickers "
          f"({(universe['market'] == 'US').sum()} US, {(universe['market'] == 'India').sum()} India)")

    print("\n" + "=" * 70)
    print("STEP 2/5: Bulk price history fetch (yfinance)")
    print("=" * 70)
    tickers = universe["ticker"].tolist()
    bulk_fetch(tickers, period=args.period, batch_size=args.batch_size,
               delay=args.delay, refresh=args.refresh)

    print("\n" + "=" * 70)
    print("STEP 3/5: Risk screen (VaR, Sharpe, volatility, drawdown -> Low/Medium/High)")
    print("=" * 70)
    risk_table = screen_universe(universe)
    risk_path = os.path.join(args.results_dir, "universe_risk_screen.csv")
    risk_table.to_csv(risk_path, index=False)
    print(f"Screened {len(risk_table)} tickers. Saved to {risk_path}")
    print(risk_table["risk_category"].value_counts())

    print("\n" + "=" * 70)
    print("STEP 4/5: Stock detail export (per-ticker JSON for the dashboard)")
    print("=" * 70)
    n_ok, n_skip = export_stock_detail_all(universe)
    print(f"Exported {n_ok} tickers ({n_skip} skipped).")

    print("\n" + "=" * 70)
    print("STEP 5/5: Pooled next-day direction model (one model, all tickers)")
    print("=" * 70)
    labeled_df, latest_df = build_pooled_dataset(universe)
    if labeled_df.empty:
        print("No usable price history for pooled model training -- skipping predictions.")
        predictions_df = pd.DataFrame(columns=["symbol", "predicted_direction", "confidence"])
        metrics_df = pd.DataFrame()
    else:
        print(f"Pooled training set: {len(labeled_df):,} rows across "
              f"{labeled_df['symbol'].nunique()} tickers.")
        metrics_df, predictions_df = train_and_evaluate_pooled(labeled_df, latest_df, args.test_frac)
        metrics_df.to_csv(os.path.join(args.results_dir, "pooled_model_metrics.csv"), index=False)
        predictions_df.to_csv(os.path.join(args.results_dir, "pooled_predictions.csv"), index=False)

    print("\n" + "=" * 70)
    print("COMBINED REPORT: risk category + predicted direction, sorted by risk")
    print("=" * 70)
    combined = risk_table.merge(
        predictions_df[["symbol", "predicted_direction", "confidence", "model_used"]] if not predictions_df.empty
        else pd.DataFrame(columns=["symbol", "predicted_direction", "confidence", "model_used"]),
        on="symbol", how="left",
    ).sort_values("risk_score", ascending=False).reset_index(drop=True)

    combined_path = os.path.join(args.results_dir, "universe_combined_report.csv")
    combined.to_csv(combined_path, index=False)

    print(combined.head(20)[["symbol", "market", "sector", "risk_category", "risk_score",
                              "predicted_direction", "confidence"]].to_string(index=False))
    print(f"\nSaved combined report ({len(combined)} tickers) to {combined_path}")

    if not metrics_df.empty:
        print("\nPooled model performance vs. naive baseline:")
        print(metrics_df[["model", "accuracy", "naive_baseline_accuracy", "beats_naive_baseline"]].to_string(index=False))


if __name__ == "__main__":
    main()
