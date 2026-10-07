"""
train_pooled_predictor.py

The ML component for the full-universe scan, done the way that actually
works at this scale: **one model trained on data pooled across every
ticker**, not one model per stock.

Why: train_predictor.py's per-stock experiment (100 days, single ticker)
showed the failure mode directly -- AAPL's models beat their naive
baseline, but GOOGL's models scored *below* theirs (both just predicted
"up" every day; there wasn't enough data for either model to learn
anything real). Pooling ~1,000 tickers' history into one training set
turns "100 rows per stock" into "hundreds of thousands of rows total",
which is what a walk-forward classifier actually needs to find a stable
signal instead of overfitting noise.

Key design points:
  - Same feature set and label definition as train_predictor.py (next-day
    up/down from today's technical features).
  - The train/test split is chronological **across the whole universe**
    (split by date, not by row count per stock) -- splitting per stock and
    then concatenating would let the model train on, say, AAPL's
    mid-2025 data while testing on RELIANCE's early-2025 data, which is
    still future leakage in spirit even if each individual stock's rows
    are in order.
  - Sector is included as a categorical feature (one-hot), on the
    hypothesis that direction-prediction signal may partly be sector-
    driven (e.g. rate-sensitive sectors behaving differently from
    defensives).
  - Evaluated the same way as the single-stock version: accuracy vs. a
    naive "always predict majority class" baseline, plus precision/
    recall/F1 -- now over a test set large enough for those numbers to be
    stable rather than noise from a 20-row test split.

Usage:
    python -m src.train_pooled_predictor                 # full cached universe
    python -m src.train_pooled_predictor --limit 20       # smoke test
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.ticker_lists import build_universe
from src.bulk_fetcher import load_cached, OUT_DIR
from src.features import build_feature_set
from src.train_predictor import FEATURE_COLUMNS, make_labels

RESULTS_PATH = "results/pooled_predictions.csv"
METRICS_PATH = "results/pooled_model_metrics.csv"


def build_pooled_dataset(universe: pd.DataFrame, data_dir: str = OUT_DIR):
    """Returns (labeled_df, latest_df): labeled_df has one row per
    (ticker, date) with a known next-day-up target, used for train/test.
    latest_df has each ticker's single most recent row (features only, no
    label yet -- tomorrow hasn't happened) used for live predictions."""
    labeled_frames, latest_frames = [], []
    skipped = 0

    for _, meta in universe.iterrows():
        ticker = meta["ticker"]
        raw = load_cached(ticker, data_dir)
        if raw is None or len(raw) < 80:
            skipped += 1
            continue

        feats = build_feature_set(raw)
        feats = make_labels(feats)
        feats["symbol"] = ticker
        feats["market"] = meta.get("market", "")
        feats["sector"] = meta.get("sector", "Unknown")

        usable = feats.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
        if usable.empty:
            skipped += 1
            continue

        latest_frames.append(usable.iloc[[-1]])  # most recent row, target may be NaN
        labeled = usable.dropna(subset=["target_next_day_up"])
        if len(labeled) >= 30:
            labeled_frames.append(labeled)

    if skipped:
        print(f"Skipped {skipped} tickers with no/insufficient cached data.")

    labeled_df = pd.concat(labeled_frames, ignore_index=True) if labeled_frames else pd.DataFrame()
    latest_df = pd.concat(latest_frames, ignore_index=True) if latest_frames else pd.DataFrame()
    return labeled_df, latest_df


def _sector_dummy_columns(df: pd.DataFrame, sector_categories: list) -> pd.DataFrame:
    sector = pd.Categorical(df["sector"], categories=sector_categories)
    return pd.get_dummies(sector, prefix="sector")


def global_walk_forward_split(df: pd.DataFrame, test_frac: float = 0.2):
    """Chronological split across the ENTIRE pooled universe: pick a cutoff
    date such that the last test_frac of the calendar range is held out,
    for every ticker at once. Splitting per-stock-then-concatenating would
    still mix eras across stocks in the pooled train set; splitting by a
    single global date cutoff keeps "the model never saw the future"
    true for the pool as a whole, not just within each stock."""
    dates = pd.to_datetime(df["date"])
    cutoff = dates.quantile(1 - test_frac)
    train_mask = dates <= cutoff
    return df[train_mask].reset_index(drop=True), df[~train_mask].reset_index(drop=True), cutoff


def train_and_evaluate_pooled(labeled_df: pd.DataFrame, latest_df: pd.DataFrame, test_frac: float = 0.2):
    sector_categories = sorted(labeled_df["sector"].fillna("Unknown").unique().tolist())

    train_df, test_df, cutoff = global_walk_forward_split(labeled_df, test_frac)
    print(f"Global chronological split at {cutoff.date()}: "
          f"{len(train_df)} train rows, {len(test_df)} test rows "
          f"({train_df['symbol'].nunique()} / {test_df['symbol'].nunique()} tickers represented).")

    def make_X(df):
        num = df[FEATURE_COLUMNS].reset_index(drop=True)
        sec = _sector_dummy_columns(df, sector_categories).reset_index(drop=True)
        return pd.concat([num, sec], axis=1)

    X_train, y_train = make_X(train_df), train_df["target_next_day_up"]
    X_test, y_test = make_X(test_df), test_df["target_next_day_up"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "logistic_regression": (LogisticRegression(max_iter=2000), X_train_scaled, X_test_scaled),
        "random_forest": (
            RandomForestClassifier(n_estimators=200, max_depth=8, n_jobs=-1, random_state=42),
            X_train, X_test,
        ),
    }

    majority_class = y_train.mode()[0]
    naive_acc = (y_test == majority_class).mean()

    metrics_rows = []
    fitted = {}
    for name, (model, Xtr, Xte) in models.items():
        model.fit(Xtr, y_train)
        fitted[name] = model
        preds = model.predict(Xte)
        acc = accuracy_score(y_test, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, preds, average="binary", zero_division=0
        )
        metrics_rows.append({
            "model": name,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "naive_baseline_accuracy": naive_acc,
            "beats_naive_baseline": acc > naive_acc,
        })
        print(f"\n=== {name} (pooled, {len(train_df):,} train rows) ===")
        print(f"  accuracy={acc:.4f}  (naive 'always predict majority' baseline: {naive_acc:.4f})")
        print(f"  precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")

    metrics_df = pd.DataFrame(metrics_rows)

    # Refit the better model (by test accuracy) on ALL labeled data, then
    # predict each ticker's latest (label-less) row for the live report.
    best_name = metrics_df.sort_values("accuracy", ascending=False).iloc[0]["model"]
    print(f"\nBest model on held-out test set: {best_name}. Refitting on all labeled data for live predictions...")

    X_all = make_X(labeled_df)
    y_all = labeled_df["target_next_day_up"]
    X_latest = _reindex_like(make_X(latest_df), X_all.columns)

    if best_name == "logistic_regression":
        final_scaler = StandardScaler()
        X_all_scaled = final_scaler.fit_transform(X_all)
        final_model = LogisticRegression(max_iter=2000)
        final_model.fit(X_all_scaled, y_all)
        X_latest_input = final_scaler.transform(X_latest)
    else:
        final_model = RandomForestClassifier(n_estimators=200, max_depth=8, n_jobs=-1, random_state=42)
        final_model.fit(X_all, y_all)
        X_latest_input = X_latest

    proba = final_model.predict_proba(X_latest_input)[:, 1]  # P(up)
    predictions_df = pd.DataFrame({
        "symbol": latest_df["symbol"].values,
        "market": latest_df["market"].values,
        "sector": latest_df["sector"].values,
        "as_of_date": latest_df["date"].values,
        "predicted_direction": np.where(proba >= 0.5, "Up", "Down"),
        "confidence": np.where(proba >= 0.5, proba, 1 - proba),
        "model_used": best_name,
    })

    return metrics_df, predictions_df


def _reindex_like(df: pd.DataFrame, columns) -> pd.DataFrame:
    return df.reindex(columns=columns, fill_value=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only use the first N tickers (smoke testing)")
    parser.add_argument("--test_frac", type=float, default=0.2)
    args = parser.parse_args()

    universe = build_universe(use_cache=True)
    if args.limit:
        universe = universe.head(args.limit)

    print(f"Building pooled dataset from up to {len(universe)} tickers...")
    labeled_df, latest_df = build_pooled_dataset(universe)

    if labeled_df.empty:
        raise SystemExit("No usable data -- run bulk_fetcher.py first.")

    print(f"Pooled dataset: {len(labeled_df):,} labeled rows across "
          f"{labeled_df['symbol'].nunique()} tickers.")

    metrics_df, predictions_df = train_and_evaluate_pooled(labeled_df, latest_df, args.test_frac)

    import os
    os.makedirs("results", exist_ok=True)
    metrics_df.to_csv(METRICS_PATH, index=False)
    predictions_df.sort_values("confidence", ascending=False).to_csv(RESULTS_PATH, index=False)

    print(f"\nSaved metrics to {METRICS_PATH}")
    print(f"Saved {len(predictions_df)} live predictions to {RESULTS_PATH}")
    print("\nTop 10 highest-confidence predictions:")
    print(predictions_df.sort_values("confidence", ascending=False).head(10).to_string(index=False))
