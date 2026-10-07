"""
export_stock_detail.py

Writes one compact JSON file per ticker (data/universe_detail/<ticker>.json)
for the dashboard's stock detail / compare views. Reuses the same tested
functions the risk screen uses -- build_feature_set (features.py),
compute_risk_report (risk_metrics.py), score_risk (risk_scoring.py) -- so
the detail view's numbers can never drift from the summary table's; there's
no separate reimplementation of the indicator math in JavaScript.

Each file has the full close-price series (for the client-side price
chart) plus the latest value of every technical indicator and risk metric
(for the stat grids) -- not the full indicator history, to keep each file
small enough to lazy-fetch on click across a ~1,000-ticker universe.

Usage:
    python -m src.export_stock_detail                # full cached universe
    python -m src.export_stock_detail --limit 20      # smoke test
"""

import argparse
import json
import os
import pandas as pd

from src.ticker_lists import build_universe
from src.bulk_fetcher import load_cached, OUT_DIR
from src.features import build_feature_set
from src.risk_metrics import compute_risk_report
from src.risk_scoring import score_risk

DETAIL_DIR = "data/universe_detail"

LATEST_FIELDS = [
    "close",
    "sma_5", "sma_10", "sma_20", "sma_50",
    "ema_5", "ema_10", "ema_20", "ema_50",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_mid", "bb_lower", "bb_width",
    "volatility_10d", "volatility_20d", "volatility_30d",
]


def _round(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return round(float(v), 4)


def export_ticker(ticker: str, meta: dict, data_dir: str = OUT_DIR):
    raw = load_cached(ticker, data_dir)
    if raw is None or len(raw) < 30:
        return None

    feats = build_feature_set(raw)
    report = compute_risk_report(feats)
    scored = score_risk(report)
    last = feats.iloc[-1]
    prev = feats.iloc[-2] if len(feats) >= 2 else None

    latest = {"date": last["date"]}
    for field in LATEST_FIELDS:
        latest[field] = _round(last.get(field))
    prev_close = _round(prev["close"]) if prev is not None else None
    latest["prev_close"] = prev_close
    if prev_close:
        latest["change"] = _round(last["close"] - prev_close)
        latest["change_pct"] = _round((last["close"] - prev_close) / prev_close)
    else:
        latest["change"] = None
        latest["change_pct"] = None

    hist = feats.dropna(subset=["close"])
    return {
        "symbol": ticker,
        "name": meta.get("name", ticker),
        "market": meta.get("market", ""),
        "sector": meta.get("sector", ""),
        "dates": hist["date"].tolist(),
        "close": [_round(v) for v in hist["close"]],
        "latest": latest,
        "risk": {k: _round(v) for k, v in report.items()},
        "risk_category": scored["risk_category"],
        "risk_score": scored["risk_score"],
    }


def export_all(universe: pd.DataFrame, out_dir: str = DETAIL_DIR, data_dir: str = OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    n_ok, n_skip = 0, 0
    for _, meta in universe.iterrows():
        payload = export_ticker(meta["ticker"], meta, data_dir)
        if payload is None:
            n_skip += 1
            continue
        with open(os.path.join(out_dir, f"{meta['ticker']}.json"), "w") as f:
            json.dump(payload, f, separators=(",", ":"))
        n_ok += 1
    print(f"Exported detail JSON for {n_ok} tickers to {out_dir}/ ({n_skip} skipped -- no/insufficient cached data).")
    return n_ok, n_skip


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only export the first N tickers (smoke testing)")
    args = parser.parse_args()

    universe = build_universe(use_cache=True)
    if args.limit:
        universe = universe.head(args.limit)

    export_all(universe)
