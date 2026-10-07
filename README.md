# Real-Time Stock Market Risk Assessment and Predictive Analytics System Using Machine Learning

Pulls stock data, computes the same quantitative risk metrics real
brokerages and robo-advisors use, assigns each stock a Low/Medium/High
risk category, and predicts next-day price direction with machine
learning -- at two scales:

- **Watchlist mode** (`src/pipeline.py`): a handful of stocks via Alpha
  Vantage, one model per stock. Good for a quick, ad-hoc check. Tested
  end-to-end on real IBM/AAPL/MSFT/GOOGL data.
- **Full-universe mode** (`src/universe_pipeline.py`): the entire S&P 500
  (US) and Nifty 500 (India) -- ~1,000 stocks -- via `yfinance`, with
  **one pooled ML model trained across all of them**. Tested end-to-end
  on real data pulled live during development, from a 20-ticker smoke
  test up through the full universe.

Both modes reuse the same tested core: `src/features.py`,
`src/risk_metrics.py`, `src/risk_scoring.py`.

## What it does

1. **Fetches** daily OHLCV price data -- Alpha Vantage for a small
   watchlist, `yfinance` (batched, cached) for the full universe.
2. **Engineers features**: returns, moving averages (SMA/EMA), RSI, MACD,
   Bollinger Bands, rolling volatility.
3. **Computes risk metrics**: historical & parametric Value at Risk (VaR),
   Sharpe ratio, max drawdown, annualized volatility, beta vs. a
   benchmark.
4. **Scores risk**: combines those metrics into an explainable Low/Medium/
   High category per stock (rule-based thresholds, not a black box --
   important when the output influences a real decision).
5. **Predicts next-day direction** (up/down):
   - *Watchlist mode*: Logistic Regression + Random Forest, one model
     per stock, walk-forward split.
   - *Full-universe mode*: **one Logistic Regression + one Random Forest
     model trained on data pooled across all ~1,000 tickers**, split
     chronologically by a single global date cutoff (not per stock), with
     each stock's sector included as a categorical feature.

## Why the full-universe model is pooled, not per-stock

The watchlist version's `train_predictor.py` was run on a 100-day IBM
sample and revealed a real, important limitation: per-stock models were
wildly inconsistent. AAPL's models meaningfully beat their naive "always
predict the majority class" baseline; GOOGL's models scored *below*
theirs (both models just predicted "up" every day -- 100 days isn't
enough data for either model to learn anything real). That's not a bug in
the code, it's what genuinely happens when you fit a model per stock on a
small sample.

Scaling to ~1,000 stocks makes "train 1,000 separate small models" *worse*,
not better -- it just runs the same small-sample problem a thousand times.
`src/train_pooled_predictor.py` instead concatenates every stock's
engineered features into one training set (hundreds of thousands of rows
instead of ~100), so the model has enough data to find a real, stable
signal rather than overfitting to one stock's noise. The tradeoff: it
learns one shared pattern across the universe rather than a
stock-specific one -- which is the right tradeoff when the alternative is
1,000 unreliable models.

## Why Alpha Vantage doesn't work at this scale

The free tier caps out at 25 requests/day, 5/minute. At one request per
stock, scanning ~1,000 stocks would take over a month. The full-universe
path uses `yfinance` instead -- no hard daily request cap, and it supports
fetching multiple tickers per call. Alpha Vantage stays wired up
separately for the small, ad-hoc "check this one stock" path, since that
already works and doesn't need to change.

## Data sources (free tier / no auth)

- **S&P 500 constituents**: `datasets/s-and-p-500-companies` on GitHub
  (`Symbol`, `Security`, `GICS Sector`, `GICS Sub-Industry`).
- **Nifty 500 constituents**: NSE India's public archives CSV
  (`Company Name`, `Industry`, `Symbol`).
- **Price history**: `yfinance` (Yahoo Finance) for the full universe;
  Alpha Vantage for the small watchlist.

Both index lists are fetched fresh each run (`src/ticker_lists.py`),
not hardcoded -- constituents change a few times a year -- and cached to
`data/universe/tickers.csv` so you're not re-fetching them on every
pipeline run. Pass `--refresh_tickers` to `universe_pipeline.py` to force
a fresh pull of the index lists themselves.

**Ticker-format handling for yfinance**, done automatically in
`src/ticker_lists.py`:
- NSE symbols get a `.NS` suffix: `RELIANCE` -> `RELIANCE.NS`.
- S&P tickers with a `.` become `-`: `BRK.B` -> `BRK-B`.

## Get a free Alpha Vantage key (watchlist mode only)

https://www.alphavantage.co/support/#api-key -- takes under 20 seconds,
no credit card.

```bash
export ALPHAVANTAGE_API_KEY=yourkeyhere
```

Full-universe mode needs no API key at all.

## Running it

```bash
pip install -r requirements.txt
```

### Watchlist mode (a few stocks, Alpha Vantage)

```bash
# offline: uses the bundled real IBM sample, no API key needed
python -m src.pipeline --offline

# live: your watchlist, needs your free API key
python -m src.pipeline --symbols AAPL MSFT GOOGL --api_key YOUR_KEY
```

### Full-universe mode (~1,000 stocks, yfinance)

```bash
# ALWAYS smoke test on a small slice first -- confirms the whole chain
# works before committing to a long fetch
python -m src.universe_pipeline --limit 20

# full S&P 500 + Nifty 500 scan
python -m src.universe_pipeline
```

Or run each stage individually:

```bash
python -m src.ticker_lists                        # fetch/cache the ~1,000-ticker universe
python -m src.bulk_fetcher --limit 20              # smoke test the fetcher
python -m src.bulk_fetcher                         # fetch price history for the full universe
python -m src.risk_screen                          # rank the full universe by risk
python -m src.train_pooled_predictor               # train the pooled model, predict next-day direction
```

**Expected runtime, honestly:**
- **First (cold) full-universe run**: fetching ~1,000 tickers' 2-year
  daily history from Yahoo Finance in batches of 75, with a courtesy
  delay between batches, takes roughly **15-30 minutes**, mostly spent
  waiting on batched network calls -- not something to run between other
  commands without expecting it to occupy that time.
- **Re-runs**: `bulk_fetcher.py` caches each ticker to
  `data/universe/<ticker>.csv` and skips anything already cached, so a
  re-run with no `--refresh` only fetches new/failed tickers -- typically
  under a minute.
- **Risk screen**: a minute or two for the full universe -- it's
  vectorized metric computation, not model training, so it scales fine.
- **Pooled model training**: a few minutes for the full universe (one
  Logistic Regression fit + one Random Forest fit over several hundred
  thousand pooled rows, plus a final refit for live predictions) -- far
  faster than training ~1,000 separate per-stock models would have been.

Failed/delisted tickers are logged and skipped, not fatal -- a scan of
~1,000 real-world tickers will always have a handful that don't resolve.

## Results

Full-universe mode writes three files to `results/`:

- `universe_risk_screen.csv` -- every screened ticker, ranked by risk
  score: symbol, market, sector/industry, risk category, risk score,
  volatility, VaR, Sharpe, max drawdown.
- `pooled_predictions.csv` -- every ticker's latest predicted next-day
  direction and confidence from the pooled model.
- `pooled_model_metrics.csv` -- pooled model accuracy/precision/recall/F1
  vs. the naive baseline, for both Logistic Regression and Random Forest.
- `universe_combined_report.csv` -- the two joined: risk category +
  predicted direction + confidence per stock, sorted by risk. This is
  what the dashboard reads.

Watchlist mode writes `results/risk_summary.csv`.

## Dashboard

`dashboard/index.html` is a small client-side app (hash-routed, no build
step, no framework) reading the `results/*.csv` files and
`data/universe_detail/<ticker>.json` files directly via `fetch`. Open it
via a local server so the browser can fetch them (`file://` will hit CORS
restrictions):

```bash
python -m http.server 8000
# then open http://localhost:8000/dashboard/
```

It has six sections, switched via the nav bar without a page reload:

- **Overview** -- hero, feature cards, sector and US-vs-India breakdowns,
  pooled-model performance, and a few computed highlights (highest-risk
  stock this scan, best pooled-model accuracy, most volatile sector) --
  all read from the actual CSV output, nothing hardcoded.
- **Risk Screen** -- the full ranked, sortable, filterable table (symbol,
  company name, market, sector, price, risk category/score, volatility,
  VaR, Sharpe, max drawdown, predicted direction). Every row is clickable
  and opens that stock's **detail view**: a header with exchange badge and
  day change (green/red), pin/compare actions, a label-value risk stat
  block, a compact sparkline that expands into a full **TradingView
  Lightweight Charts** view (real pan/zoom + a crosshair showing exact
  date/price, with 1W/1M/3M/1Y/Max range toggles), every technical
  indicator from `features.py` with a plain-language read, and a **News
  sentiment** card (see below). From there, **Compare** opens a
  search-and-pick control that defaults to a ranked **related-stocks**
  list -- same sector/industry (either market), ranked by Pearson
  correlation of daily returns, computed lazily and cached per session --
  and puts two stocks' detail views side by side.
- **Watchlist** -- pin any stock (from the table's star icon or a detail
  view's pin button) to track it here; persisted in the browser's
  `localStorage`.
- **Portfolio** -- your real holdings, see below.
- **Methodology** -- the honesty notes below, in prose, for the write-up.

The detail/compare views need one JSON file per ticker
(`data/universe_detail/<ticker>.json`, produced by
`src/export_stock_detail.py`, wired into `universe_pipeline.py` as step
4/5) -- it reuses `build_feature_set` and `compute_risk_report` exactly,
so the detail view's numbers can't drift from the summary table's. If you
run the pipeline stages individually rather than through
`universe_pipeline.py`, run this after `bulk_fetcher.py`:

```bash
python -m src.export_stock_detail
```

### Portfolio

Add real holdings two ways, both `localStorage`-persisted, no backend:

- **Manual add**: search the universe (same control as Compare), enter
  quantity + average buy price.
- **CSV import**: upload a holdings export from Zerodha Console, INDmoney,
  or similar. The parser matches common column-name variants (`Symbol` /
  `Instrument` / `Ticker`, `Qty` / `Quantity`, `Avg. cost` / `Average
  Price`) rather than requiring one exact format, and normalizes bare NSE
  symbols (`RELIANCE`) to our `.NS` ticker form. Rows it can't match are
  reported, not silently dropped. Sample exports to try it with:
  `data/sample/zerodha_holdings_sample.csv`,
  `data/sample/indmoney_holdings_sample.csv`.

The **Portfolio** page then computes, from whatever's entered:

- **Portfolio-level risk, done properly** -- not a naive average of each
  holding's own volatility. Builds an actual covariance matrix of daily
  returns across holdings (same date-alignment logic as the related-stocks
  correlation, ported into `dashboard/index.html`) and computes
  `portfolio_variance = wᵀ·Cov·w` for annualized volatility, plus
  historical VaR/Sharpe/max drawdown from the realized weighted daily
  portfolio return series -- all using the same formulas as
  `risk_metrics.py`.
- **Concentration**: sector breakdown by value, with an explicit flag when
  any single holding or sector exceeds 25%.
- **Individual risk flags**: any holding independently categorized High
  risk in the main risk screen.
- **Aggregate sentiment**: value-weighted average of each holding's
  sentiment score (see below), with a coverage note when not every holding
  has cached sentiment data yet.

Kite Connect live sync (mentioned as an optional, advanced path in the
original spec) was deliberately **not built** -- it requires your own paid
Kite Connect subscription, and this project sticks to free-tier data only.
Manual add and CSV import cover the same ground without a subscription.

### News sentiment

`src/sentiment.py` fetches recent news + Alpha Vantage's own per-article
sentiment score per ticker (`NEWS_SENTIMENT`, same free API key as
`data_fetcher.py`; add a real key with `--api_key` or
`ALPHAVANTAGE_API_KEY`), aggregates the last ~20-50 articles into one score
(-1..+1, plus Alpha Vantage's own Bearish..Bullish label bands), and caches
the result to `data/sentiment/<ticker>.json` -- same lazy, per-ticker
caching pattern as everything else in `data/`.

```bash
python -m src.sentiment --symbols AAPL MSFT --api_key YOUR_KEY
```

Optionally, also scores the same headlines with a real pretrained
financial NLP model -- `ProsusAI/finbert` via HuggingFace `transformers`
-- as an independent second opinion rather than relying solely on Alpha
Vantage's number:

```bash
python -m src.sentiment --symbols AAPL --api_key YOUR_KEY --finbert
```

This is opt-in because it downloads a real ~400MB model on first use;
without the flag, or without `transformers` installed, it falls back
cleanly to the Alpha Vantage score alone -- the dashboard shows "not run,
optional" rather than pretending a score exists. `transformers` was not
installed for this project's development environment, so the FinBERT path
is implemented and exercised by its ImportError fallback, but not run
end-to-end here.

**Testing note**: unlike `TIME_SERIES_DAILY` (whose free `demo` key
returns real IBM data), Alpha Vantage's `demo` key returns no real data
for `NEWS_SENTIMENT` for any symbol -- there's no way to validate a live
response without your own free key. `src/sentiment.py`'s parsing and
aggregation logic is instead validated against a hand-built, realistic
sample response matching Alpha Vantage's documented schema:

```bash
python -m src.sentiment --symbols AAPL --use_sample   # no API key needed
```

The dashboard shows the score, label, and the actual recent headlines with
their individual scores on each stock's detail view (not just a bare
number), plus the value-weighted aggregate on the Portfolio page.

## Honesty notes (read before writing the paper/pitch)

- **"Real-time"**: this means "refreshed on demand from the freshest free
  data available" (daily bars), not live tick-by-tick streaming. Say this
  plainly -- it's still a real, useful system, just don't overclaim.
- **Direction prediction is genuinely hard**: markets are close to
  efficient at a 1-day horizon, so don't expect near-100% accuracy. A
  model that consistently beats the naive baseline by even a few points,
  evaluated correctly (walk-forward split, not random shuffle), is a
  legitimate and defensible result.
- **Small-sample instability is a real finding, not just a bug**: the
  single-stock experiment (AAPL beating its baseline, GOOGL's models
  stuck predicting one class) is genuine evidence that per-stock models
  don't have enough data to be trustworthy at this horizon. The pooled
  model exists specifically to address that, and the honest comparison
  between the two approaches (per-stock instability vs. pooled stability)
  is a stronger result to report than either number alone.
- **Risk thresholds**: the Low/Medium/High cutoffs in `risk_scoring.py`
  are reasonable starting points, not gospel. Justify them explicitly
  (e.g. against a benchmark like SPY) when you write this up.
- **Free-tier data only**: no paid APIs anywhere in this project -- Kite
  Connect live sync was deliberately left unbuilt for exactly this reason.
- **Mixed-currency portfolios**: if your Portfolio holdings span both US
  ($) and India (₹), the blended risk/concentration weights use raw price
  × quantity with no FX conversion (no FX data source in this project).
  Per-market subtotals are shown separately and are exact; the blended
  numbers are directional, not precise, and the dashboard says so inline.
- **`NEWS_SENTIMENT` was never live-tested**: Alpha Vantage's free `demo`
  key doesn't return real data for this endpoint for any symbol, and no
  paid/personal key was available during development. The fetch and
  aggregation code is written against Alpha Vantage's documented response
  shape and validated against a hand-built realistic sample -- it should
  work with your own free key, but say so honestly rather than claiming
  it was verified against live data, unlike the price endpoints.

## Project structure

```
stock-risk-ml/
  data/
    sample/                  # real IBM sample, sample broker CSVs, sample sentiment response
    live/                    # watchlist-mode live-fetched data lands here
    universe/                # full-universe: tickers.csv + one CSV per ticker
    universe_detail/          # per-ticker JSON for the dashboard's detail/compare views
    sentiment/                 # per-ticker cached news sentiment (src/sentiment.py)
  src/
    data_fetcher.py          # Alpha Vantage API calls (watchlist mode + news sentiment)
    features.py               # technical indicators (shared)
    risk_metrics.py           # VaR, Sharpe, drawdown, beta (shared)
    risk_scoring.py           # Low/Medium/High risk categorization (shared)
    train_predictor.py        # per-stock next-day direction ML model (watchlist mode)
    pipeline.py                # watchlist mode entry point
    ticker_lists.py            # S&P 500 + Nifty 500 constituent fetch/normalize/cache
    bulk_fetcher.py            # batched yfinance fetch for the full universe
    risk_screen.py             # risk screen across the full universe
    export_stock_detail.py     # per-ticker detail JSON for the dashboard
    train_pooled_predictor.py  # pooled ML model across the full universe
    sentiment.py                # news sentiment: Alpha Vantage + optional FinBERT
    universe_pipeline.py       # full-universe mode entry point
  dashboard/
    index.html                 # hash-routed dashboard: Overview/Risk Screen/detail/compare/Watchlist/Portfolio/Methodology
  results/                     # risk_summary.csv / universe_*.csv land here after each run
  requirements.txt
```
