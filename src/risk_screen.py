"""
risk_screen.py

Runs the existing, tested risk pipeline (features.py -> risk_metrics.py ->
risk_scoring.py) across every ticker cached by bulk_fetcher.py, producing
one ranked table for the full universe. No ML training here -- just the
feature/metric computation, so this is fast enough to run on the full
~1,000-stock universe every time the data refreshes.

Usage:
    python -m src.risk_screen                  # full cached universe
    python -m src.risk_screen --limit 20        # smoke test
"""

import argparse
import pandas as pd

from src.ticker_lists import build_universe
from src.bulk_fetcher import load_cached, OUT_DIR
from src.features import build_feature_set
from src.risk_metrics import compute_risk_report
from src.risk_scoring import score_risk

RESULTS_PATH = "results/universe_risk_screen.csv"


def screen_universe(universe: pd.DataFrame, data_dir: str = OUT_DIR) -> pd.DataFrame:
    rows = []
    missing = 0
    for _, meta in universe.iterrows():
        ticker = meta["ticker"]
        raw = load_cached(ticker, data_dir)
        if raw is None or len(raw) < 30:
            missing += 1
            continue

        feats = build_feature_set(raw)
        report = compute_risk_report(feats)
        scored = score_risk(report)

        rows.append({
            "symbol": ticker,
            "raw_symbol": meta.get("raw_symbol", ticker),
            "name": meta.get("name", ""),
            "market": meta.get("market", ""),
            "sector": meta.get("sector", ""),
            "industry": meta.get("industry", ""),
            "risk_category": scored["risk_category"],
            "risk_score": scored["risk_score"],
            "annualized_volatility": report["annualized_volatility"],
            "historical_var_95": report["historical_var_95"],
            "parametric_var_95": report["parametric_var_95"],
            "sharpe_ratio": report["sharpe_ratio"],
            "max_drawdown": report["max_drawdown"],
            "last_close": raw["close"].iloc[-1],
            "last_date": raw["date"].iloc[-1],
        })

    if missing:
        print(f"Skipped {missing} tickers with no/insufficient cached data "
              f"(run bulk_fetcher.py first).")

    table = pd.DataFrame(rows).sort_values("risk_score", ascending=False).reset_index(drop=True)
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="only screen the first N tickers (smoke testing)")
    parser.add_argument("--results_path", type=str, default=RESULTS_PATH)
    args = parser.parse_args()

    universe = build_universe(use_cache=True)
    if args.limit:
        universe = universe.head(args.limit)

    print(f"Screening {len(universe)} tickers...")
    table = screen_universe(universe)

    import os
    os.makedirs(os.path.dirname(args.results_path), exist_ok=True)
    table.to_csv(args.results_path, index=False)

    print(f"\nScreened {len(table)} tickers successfully.")
    print(table["risk_category"].value_counts())
    print("\nTop 10 highest risk:")
    print(table.head(10)[["symbol", "market", "sector", "risk_category", "risk_score",
                           "annualized_volatility", "sharpe_ratio"]].to_string(index=False))
    print(f"\nSaved to {args.results_path}")
