"""
bulk_fetcher.py

Fetches daily OHLCV history for the full S&P 500 + Nifty 500 universe via
`yfinance`. Alpha Vantage's free tier (25 requests/day) can't cover ~1,000
tickers -- yfinance has no hard daily cap and supports multi-ticker batch
downloads, which is what makes scanning the full universe practical.

Each ticker's history is cached to data/universe/<ticker>.csv so re-runs
only fetch what's missing. Pass --refresh to force re-fetching everything.

Batches are fetched with a short delay between them (--delay, default 2s)
to be a reasonable citizen of Yahoo's free, unofficial endpoint rather than
firing 1,000 individual requests back to back.

Usage:
    # smoke test on a handful of tickers first
    python -m src.bulk_fetcher --limit 20

    # full universe
    python -m src.bulk_fetcher

    # force re-fetch everything
    python -m src.bulk_fetcher --refresh
"""

import argparse
import os
import time
import pandas as pd
import yfinance as yf

from src.ticker_lists import build_universe

OUT_DIR = "data/universe"


def _ticker_cache_path(ticker: str, out_dir: str = OUT_DIR) -> str:
    # yfinance tickers can contain "-" (BRK-B) but not path-unsafe chars, safe as filenames
    return os.path.join(out_dir, f"{ticker}.csv")


def _clean_ticker_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """raw: single-ticker slice of a yf.download(group_by='ticker') result,
    columns like Open/High/Low/Close/Volume, indexed by Date."""
    df = raw.dropna(how="all").copy()
    if df.empty:
        return df
    df = df.rename(columns={c: c.lower() for c in df.columns})
    df = df.reset_index().rename(columns={"Date": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    keep = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]
    return df[keep].dropna(subset=["close"]).reset_index(drop=True)


def fetch_batch(tickers: list, period: str = "2y") -> dict:
    """Returns {ticker: DataFrame}, only for tickers that returned usable data."""
    try:
        raw = yf.download(
            tickers=tickers, period=period, group_by="ticker",
            threads=True, progress=False, auto_adjust=True,
        )
    except Exception as e:
        print(f"  Batch download failed outright: {e}")
        return {}

    out = {}
    # yf.download returns a flat (non-multiindex) frame when len(tickers) == 1
    if len(tickers) == 1:
        cleaned = _clean_ticker_frame(raw)
        if not cleaned.empty:
            out[tickers[0]] = cleaned
        return out

    for ticker in tickers:
        if ticker not in raw.columns.get_level_values(0):
            continue
        cleaned = _clean_ticker_frame(raw[ticker])
        if not cleaned.empty:
            out[ticker] = cleaned
    return out


def bulk_fetch(tickers: list, out_dir: str = OUT_DIR, period: str = "2y",
                batch_size: int = 50, delay: float = 2.0, refresh: bool = False) -> dict:
    os.makedirs(out_dir, exist_ok=True)

    if not refresh:
        pending = [t for t in tickers if not os.path.exists(_ticker_cache_path(t, out_dir))]
        skipped_cached = len(tickers) - len(pending)
        if skipped_cached:
            print(f"{skipped_cached} tickers already cached, skipping (use --refresh to force).")
    else:
        pending = list(tickers)

    succeeded, failed = [], []
    batches = [pending[i:i + batch_size] for i in range(0, len(pending), batch_size)]
    print(f"Fetching {len(pending)} tickers in {len(batches)} batch(es) of up to {batch_size}...")

    for i, batch in enumerate(batches):
        print(f"Batch {i+1}/{len(batches)} ({len(batch)} tickers)...")
        results = fetch_batch(batch, period=period)
        for ticker in batch:
            df = results.get(ticker)
            if df is None or df.empty:
                failed.append(ticker)
                continue
            df.to_csv(_ticker_cache_path(ticker, out_dir), index=False)
            succeeded.append(ticker)
        if i < len(batches) - 1:
            time.sleep(delay)

    if failed:
        print(f"\n{len(failed)} tickers failed/had no data, skipped: "
              f"{', '.join(failed[:20])}{' ...' if len(failed) > 20 else ''}")
    print(f"Fetched {len(succeeded)} new tickers this run.")

    all_cached = {t for t in tickers if os.path.exists(_ticker_cache_path(t, out_dir))}
    return {"succeeded": succeeded, "failed": failed, "cached_total": len(all_cached)}


def load_cached(ticker: str, out_dir: str = OUT_DIR) -> pd.DataFrame:
    path = _ticker_cache_path(ticker, out_dir)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="only fetch the first N tickers from the universe (for smoke testing)")
    parser.add_argument("--tickers", nargs="+", default=None,
                         help="explicit ticker list, overrides --limit / the full universe")
    parser.add_argument("--period", type=str, default="2y",
                         help="yfinance period, e.g. 1y, 2y, 5y, max")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--delay", type=float, default=2.0, help="seconds to pause between batches")
    parser.add_argument("--refresh", action="store_true", help="force re-fetch even if cached")
    args = parser.parse_args()

    if args.tickers:
        tickers = args.tickers
    else:
        universe = build_universe(use_cache=True)
        tickers = universe["ticker"].tolist()
        if args.limit:
            tickers = tickers[:args.limit]

    summary = bulk_fetch(
        tickers, period=args.period, batch_size=args.batch_size,
        delay=args.delay, refresh=args.refresh,
    )
    print(f"\nDone. {summary['cached_total']} tickers now cached in {OUT_DIR}/")
