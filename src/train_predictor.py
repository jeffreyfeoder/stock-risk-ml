"""
train_predictor.py

The ML component: predicts whether tomorrow's close will be higher than
today's (binary up/down classification) from today's engineered technical
features. This is deliberately framed as direction prediction, not exact
price prediction -- predicting an exact future price is a much harder
(often misleading) problem, while direction is a well-posed classification
task and is what a lot of real trading-signal research actually targets.

Be upfront in any write-up: stock direction prediction from technical
features alone is genuinely hard (markets are close to efficient at this
horizon), so don't expect or claim near-100% accuracy. A model that beats
50-55% consistently, with the right evaluation (walk-forward, not random
shuffle), is a legitimate result -- and the honest story ("here's why this
is hard, here's how much signal actually exists") is a better paper than
overclaiming.

Usage:
    python -m src.train_predictor data/sample/IBM_daily.csv
"""

import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

from src.features import build_feature_set


FEATURE_COLUMNS = [
    "daily_return", "log_return",
    "sma_5", "sma_10", "sma_20",
    "ema_5", "ema_10", "ema_20",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_width", "volatility_10d", "volatility_20d",
]


def make_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Label = 1 if tomorrow's close > today's close, else 0. Uses shift(-1)
    so the label is genuinely about the future relative to that row's
    features -- drop the last row afterward since it has no "tomorrow"."""
    df = df.copy()
    df["target_next_day_up"] = (df["close"].shift(-1) > df["close"]).astype(int)
    return df


def walk_forward_split(df: pd.DataFrame, test_frac: float = 0.2):
    """Chronological split -- train on the past, test on the future. A
    random split would leak information backwards in time and overstate
    performance, same issue as with the fraud detection project."""
    n = len(df)
    cutoff = int(n * (1 - test_frac))
    return df.iloc[:cutoff], df.iloc[cutoff:]


def train_and_evaluate(csv_path: str):
    raw = pd.read_csv(csv_path)
    df = build_feature_set(raw)
    df = make_labels(df)
    df = df.dropna(subset=FEATURE_COLUMNS + ["target_next_day_up"]).reset_index(drop=True)

    if len(df) < 30:
        print(f"WARNING: only {len(df)} usable rows after feature warm-up and label shift. "
              f"This sample (100 raw days) is enough to prove the pipeline runs, but far too "
              f"little data for a real result -- fetch --outputsize full (20+ years) for the "
              f"real project.")

    train_df, test_df = walk_forward_split(df)
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["target_next_day_up"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["target_next_day_up"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "logistic_regression": (LogisticRegression(max_iter=2000), X_train_scaled, X_test_scaled),
        "random_forest": (RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42), X_train, X_test),
    }

    results = {}
    for name, (model, Xtr, Xte) in models.items():
        model.fit(Xtr, y_train)
        preds = model.predict(Xte)
        acc = accuracy_score(y_test, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, preds, average="binary", zero_division=0
        )
        # naive baseline: always predict the majority class in the training set
        majority_class = y_train.mode()[0]
        naive_acc = (y_test == majority_class).mean()

        results[name] = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "naive_baseline_accuracy": naive_acc,
            "beats_naive_baseline": acc > naive_acc,
        }
        print(f"\n=== {name} ===")
        print(f"  accuracy={acc:.4f}  (naive 'always predict majority' baseline: {naive_acc:.4f})")
        print(f"  precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")

    return results


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample/IBM_daily.csv"
    train_and_evaluate(path)
