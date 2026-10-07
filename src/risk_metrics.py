"""
risk_metrics.py

Standard quantitative risk metrics, computed from daily returns. These are
the same core metrics real brokerages, robo-advisors, and portfolio
management tools use -- nothing exotic, which is exactly why they're
defensible in a report or pitch.

Metrics:
  - Historical VaR (Value at Risk): "on the worst 5% of days, how much do
    you lose?" -- read directly off the empirical return distribution.
  - Parametric VaR: same question, assuming returns are normally
    distributed (faster to compute, less accurate in fat-tailed markets --
    reporting both and noting the gap is a good discussion point).
  - Sharpe ratio: return earned per unit of risk taken (annualized).
  - Max drawdown: the worst peak-to-trough decline over the period --
    what an investor holding through the worst stretch would have felt.
  - Beta: how much the stock moves relative to a benchmark (e.g. SPY).
"""

import numpy as np
import pandas as pd


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Returns a positive number representing the loss threshold, e.g.
    0.032 means "5% of days lost more than 3.2%"."""
    returns = returns.dropna()
    if len(returns) == 0:
        return float("nan")
    return -np.percentile(returns, (1 - confidence) * 100)


def parametric_var(returns: pd.Series, confidence: float = 0.95) -> float:
    from scipy.stats import norm
    returns = returns.dropna()
    if len(returns) == 0:
        return float("nan")
    mu, sigma = returns.mean(), returns.std()
    z = norm.ppf(1 - confidence)
    return -(mu + z * sigma)


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.04) -> float:
    """Annualized Sharpe ratio. risk_free_rate is annual (default ~4%,
    roughly a recent T-bill rate -- update to whatever's current when you
    write this up)."""
    returns = returns.dropna()
    if len(returns) < 2 or returns.std() == 0:
        return float("nan")
    daily_rf = risk_free_rate / 252
    excess = returns - daily_rf
    return (excess.mean() / returns.std()) * np.sqrt(252)


def max_drawdown(close_prices: pd.Series) -> float:
    """Returns a positive number: the largest peak-to-trough % decline."""
    prices = close_prices.dropna()
    if len(prices) == 0:
        return float("nan")
    running_max = prices.cummax()
    drawdown = (prices - running_max) / running_max
    return -drawdown.min()


def beta(stock_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    aligned = pd.concat([stock_returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return float("nan")
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1]
    var = np.var(aligned.iloc[:, 1])
    return cov / var if var != 0 else float("nan")


def annualized_volatility(returns: pd.Series) -> float:
    returns = returns.dropna()
    if len(returns) == 0:
        return float("nan")
    return returns.std() * np.sqrt(252)


def compute_risk_report(df: pd.DataFrame, benchmark_returns: pd.Series = None) -> dict:
    """df must have a 'daily_return' and 'close' column (see features.py)."""
    returns = df["daily_return"]
    report = {
        "annualized_volatility": annualized_volatility(returns),
        "historical_var_95": historical_var(returns, 0.95),
        "historical_var_99": historical_var(returns, 0.99),
        "parametric_var_95": parametric_var(returns, 0.95),
        "sharpe_ratio": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(df["close"]),
    }
    if benchmark_returns is not None:
        report["beta"] = beta(returns, benchmark_returns)
    return report


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.features import build_feature_set

    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample/IBM_daily.csv"
    df = pd.read_csv(path)
    df = build_feature_set(df)
    report = compute_risk_report(df)
    for k, v in report.items():
        print(f"{k}: {v:.4f}")
