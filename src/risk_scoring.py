"""
risk_scoring.py

Turns the raw risk metrics from risk_metrics.py into a single, explainable
Low/Medium/High risk category per stock -- the kind of summary a person
actually wants to glance at, versus six separate numbers.

Deliberately rule-based (thresholds), not a black-box ML model: for a risk
score that influences real decisions, explainability matters more than
squeezing out extra accuracy, and thresholds are trivial to justify in a
paper/pitch ("we flag High risk when annualized volatility exceeds 40%,
which is roughly double the long-run S&P 500 average").

Thresholds below are reasonable starting points, not gospel -- tune them
against a benchmark (e.g. compute the same metrics for SPY) and justify
your choice explicitly when you write this up.
"""

import pandas as pd
from src.risk_metrics import compute_risk_report


DEFAULT_THRESHOLDS = {
    "annualized_volatility": {"medium": 0.30, "high": 0.50},   # 30% / 50% annualized vol
    "historical_var_95": {"medium": 0.025, "high": 0.045},      # 2.5% / 4.5% daily VaR
    "max_drawdown": {"medium": 0.20, "high": 0.35},             # 20% / 35% peak-to-trough
}


def _score_metric(value: float, thresholds: dict) -> int:
    """Returns 0 (low), 1 (medium), 2 (high) for a single metric."""
    if pd.isna(value):
        return 0
    if value >= thresholds["high"]:
        return 2
    if value >= thresholds["medium"]:
        return 1
    return 0


def score_risk(report: dict, thresholds: dict = None) -> dict:
    thresholds = thresholds or DEFAULT_THRESHOLDS
    sub_scores = {
        metric: _score_metric(report[metric], thresholds[metric])
        for metric in thresholds
        if metric in report
    }
    avg_score = sum(sub_scores.values()) / len(sub_scores) if sub_scores else 0
    if avg_score >= 1.5:
        category = "High"
    elif avg_score >= 0.5:
        category = "Medium"
    else:
        category = "Low"
    return {
        "risk_category": category,
        "risk_score": round(avg_score, 2),
        "sub_scores": sub_scores,
        "metrics": report,
    }


def score_watchlist(feature_dfs: dict, thresholds: dict = None) -> pd.DataFrame:
    """feature_dfs: {symbol: DataFrame with daily_return/close columns already computed}"""
    rows = []
    for symbol, df in feature_dfs.items():
        report = compute_risk_report(df)
        scored = score_risk(report, thresholds)
        rows.append({
            "symbol": symbol,
            "risk_category": scored["risk_category"],
            "risk_score": scored["risk_score"],
            "annualized_volatility": report["annualized_volatility"],
            "historical_var_95": report["historical_var_95"],
            "sharpe_ratio": report["sharpe_ratio"],
            "max_drawdown": report["max_drawdown"],
        })
    return pd.DataFrame(rows).sort_values("risk_score", ascending=False)


if __name__ == "__main__":
    import sys
    from src.features import build_feature_set

    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample/IBM_daily.csv"
    symbol = sys.argv[2] if len(sys.argv) > 2 else "IBM"

    df = pd.read_csv(path)
    df = build_feature_set(df)
    result = score_watchlist({symbol: df})
    print(result.to_string(index=False))
