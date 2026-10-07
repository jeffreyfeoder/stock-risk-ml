# Master prompt for Claude Code

Paste everything between the lines below into Claude Code in VS Code,
opened at the root of this `stock-risk-ml/` folder. It already contains a
working, tested foundation (`src/data_fetcher.py`, `src/features.py`,
`src/risk_metrics.py`, `src/risk_scoring.py`, `src/train_predictor.py`,
`src/pipeline.py`) that was run end-to-end on real IBM/AAPL/MSFT/GOOGL
data pulled live from Alpha Vantage -- point Claude Code at this folder so
it reads the existing files before it starts.

---

I'm scaling up my project, **Real-Time Stock Market Risk Assessment and
Predictive Analytics System Using Machine Learning**, from a small
watchlist to the full S&P 500 (US) and Nifty 500 (India) -- around 1,000
major, liquid stocks covering the large majority of both markets' total
value. This is a project-based-learning assignment; I need the working
implementation, not just a plan.

## What's already built and tested (don't rewrite unless something's wrong)

- `src/data_fetcher.py` -- fetches daily OHLCV data from Alpha Vantage
  for a small watchlist. **Tested against real IBM data.** Keep this as-is
  for the "quick check a few stocks" use case -- it's fine for that, just
  not for 1,000 stocks (see below).
- `src/features.py` -- technical indicators (SMA/EMA, RSI, MACD, Bollinger
  Bands, rolling volatility). **Tested.**
- `src/risk_metrics.py` -- VaR (historical + parametric), Sharpe ratio,
  max drawdown, annualized volatility, beta. **Tested.**
- `src/risk_scoring.py` -- turns risk metrics into a Low/Medium/High
  category per stock (rule-based thresholds). **Tested.**
- `src/train_predictor.py` -- ML next-day direction classifier
  (Logistic Regression + Random Forest), one model per stock, walk-forward
  split. **Tested and revealing an important lesson**: on a 100-day
  sample, per-stock models were wildly inconsistent -- AAPL beat its naive
  baseline meaningfully, but GOOGL's models scored *below* their naive
  baseline (both models just predicted "up" every day -- not enough data
  to learn anything real). **This is why the full-universe version below
  needs a different ML approach, not just "run the same thing 1,000
  times."**
- `src/pipeline.py` -- orchestrates fetch -> features -> risk -> score ->
  predict for a small watchlist. **Tested.**

## Why Alpha Vantage doesn't work for this scale

The free tier caps out at 25 requests/day, 5/minute. At one request per
stock, scanning ~1,000 stocks would take over a month. **Switch the bulk
data source to `yfinance`** (Yahoo Finance via the `yfinance` pip
package) for the full-universe fetch -- no hard daily request cap, and it
supports fetching multiple tickers per call (`yf.download(tickers=[...])`).
Keep Alpha Vantage wired up separately for the small, ad-hoc "check this
one stock" path since that already works.

## Ticker lists (verified working, no auth needed)

- **S&P 500**: `https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv`
  (columns include `Symbol`, `Security`, `GICS Sector`). Verified this
  returns real, current data.
- **Nifty 500**: `https://archives.nseindia.com/content/indices/ind_nifty500list.csv`
  (columns: `Company Name,Industry,Symbol,Series,ISIN Code`). Verified this
  returns real, current data (fetched during development -- ~500 rows,
  e.g. RELIANCE, TCS, HDFCBANK, INFY, etc.).

Fetch these fresh each time rather than hardcoding a snapshot -- index
constituents change periodically (additions/removals happen a few times a
year), so the list should always be current when the pipeline runs.

**Important ticker-format gotchas for yfinance:**
- NSE (Nifty 500) symbols need a `.NS` suffix for yfinance, e.g.
  `RELIANCE` -> `RELIANCE.NS`.
- A handful of S&P 500 tickers have dots that need converting to dashes
  for yfinance, e.g. `BRK.B` -> `BRK-B`.

## What I need you to build

1. **`src/ticker_lists.py`** -- fetches both index constituent lists from
   the URLs above, normalizes symbols for yfinance (the `.NS` suffix /
   dash conversion above), caches the combined list to
   `data/universe/tickers.csv` with a `market` column (`US` or `India`)
   so it's not re-fetched every run.

2. **`src/bulk_fetcher.py`** -- fetches daily OHLCV history for the full
   ~1,000-ticker universe via `yfinance`, in batches (yfinance supports
   multi-ticker downloads, and batching avoids hammering Yahoo's endpoint
   -- add a small delay between batches to be a reasonable API citizen).
   Cache each ticker's data to `data/universe/<ticker>.csv` so re-runs
   don't re-fetch everything; add a `--refresh` flag to force re-fetch.
   Handle failures gracefully (delisted tickers, temporary fetch errors)
   -- log and skip, don't crash the whole run.

3. **`src/risk_screen.py`** -- runs `features.py` + `risk_metrics.py` +
   `risk_scoring.py` (reuse these as-is, they're tested) across every
   ticker in the universe, producing one big ranked table: symbol, market,
   sector/industry, risk category, risk score, volatility, VaR, Sharpe,
   max drawdown. Save to `results/universe_risk_screen.csv`. This should
   be fast (no ML training involved, just the vectorized metrics), so it's
   fine to run on the full ~1,000 stocks every time.

4. **`src/train_pooled_predictor.py`** -- the ML component, done the
   scalable and statistically sound way: **one model trained on pooled
   data across all ~1,000 stocks** (not one model per stock -- that's what
   fell apart in testing). Concatenate each stock's engineered features
   into one big training set, with the same walk-forward chronological
   split logic (split by date across the whole universe, not per stock,
   so there's no future leakage). Consider adding the stock's sector as a
   categorical feature. Evaluate the same way `train_predictor.py` did:
   accuracy vs. the naive baseline, precision/recall/F1 -- and now with
   ~1,000x the training data, these numbers should be far more stable
   than the single-stock experiment was.

5. **`src/universe_pipeline.py`** -- the new top-level entry point:
   ticker lists -> bulk fetch -> risk screen -> pooled prediction ->
   combined report (risk category + predicted direction + confidence per
   stock, sorted by risk). Keep `src/pipeline.py` (the small-watchlist
   version) working independently -- both should coexist.

6. Update `README.md` with the new scale: how to run the full-universe
   scan, expected runtime (be honest -- fetching ~1,000 tickers will take
   a while even with yfinance, estimate and state it), and results.

## Constraints and honesty notes (carry these over, they matter for the write-up)

- **"Real-time"** = refreshed on demand from the freshest free data
  available, not live tick streaming. Don't let generated docs imply
  otherwise.
- **Small-sample instability is a real finding, not just a bug** -- the
  per-stock experiment earlier showed a model can look fine on one stock
  and be objectively broken (always predicting one class) on another with
  the same code and settings. The pooled-model approach in step 4 exists
  specifically to fix this; mention the comparison in whatever
  writeup/report gets generated.
- Free-tier/no-auth data sources only -- no paid APIs.
- After building each piece, actually run it and show me the output
  (including on a *small* subset first, e.g. 20 tickers, before doing the
  full ~1,000 -- yfinance fetches for 1,000 tickers will take real time
  and you don't want to debug a bug after a 20-minute fetch).

Start with `src/ticker_lists.py` and `src/bulk_fetcher.py` (test on ~20
tickers first), then `src/risk_screen.py`, then the pooled predictor, then
wire it all together in `src/universe_pipeline.py`.

## 7. Dashboard UI

Once the pipeline produces `results/universe_risk_screen.csv` and the
pooled model's predictions, build a single-page web dashboard
(`dashboard/index.html` -- plain HTML/CSS/JS is fine, or Streamlit if that's
faster to wire up to the CSVs, your call) that visualizes it. I want the
visual style to match a specific reference design -- a bold black/white/
neon-green agency-style landing page. Here's the exact spec, translated
from that reference into a data-dashboard layout:

**Color palette**: near-black background sections (`#0a0a0a` or similar)
alternating with white/off-white sections. One vivid accent color -- a
lime/chartreuse green (`#CFFF04`-ish) -- used for buttons, highlight
badges, and risk-positive indicators. A muted forest green as a secondary
accent for smaller icon details. Body text white-on-black or black-on-white
depending on section; avoid any other colors creeping in outside the
risk-category color coding described below.

**Typography**: large, bold, geometric sans-serif headings (Inter,
Space Grotesk, or similar), generous letter-spacing on nav/labels, clean
medium-weight body text. Headlines should be short and punchy.

**Layout, top to bottom**:

1. **Nav bar**: logo/project name top-left, nav links centered for each
   real section of the app -- **Overview, Risk Screen, Watchlist,
   Methodology** (Overview = the hero/stats landing view, Risk Screen =
   the full stock table, Watchlist = pinned stocks, Methodology = a plain
   page explaining the risk metrics/model in prose, useful for the paper
   later) -- plus a rounded black pill button top-right ("Refresh data" or
   "Run scan"). This should behave like a real multi-section app: clicking
   a nav link switches the visible section without a full page reload
   (client-side show/hide or client-side routing, your call).

2. **Hero section**: large bold 2-line headline on the left (e.g. "Scan
   1,000 stocks. Know the risk before the market does." -- write your own,
   keep it punchy), one line of subtext, a rounded black CTA pill button
   ("View risk screen"). On the right, an abstract illustration in the
   same line-art style as the reference (small circular badges with icons
   -- swap the reference's megaphone/social icons for finance-relevant
   ones: an upward arrow, a candlestick/bar-chart glyph, a shield icon for
   "risk" -- floating around a central abstract swirl shape). SVG is fine,
   keep it simple line art, not a literal chart.

3. **Trust/data-source strip**: a row of small grayscale labels for the
   real data sources this project actually uses -- "S&P 500", "Nifty 500",
   "Yahoo Finance", "NSE India" -- styled like a logo strip even though
   they're just text/wordmarks.

4. **Section label + 2x2 (or more) feature card grid**: a small pill-
   shaped label (e.g. "Features") above a section heading, then a grid of
   rounded cards alternating black-background/white-background (like the
   reference's four service cards), each with: a small icon, a bold title,
   one line of description, and a "Learn more →" style link at the
   bottom-left with a small circular arrow badge. Use this grid for the
   dashboard's actual sections: "Risk Screening" (links to/summarizes the
   full ranked table), "Direction Prediction" (summarizes pooled model
   accuracy), "Sector Breakdown" (risk by GICS/industry sector), "Market
   Comparison" (US vs. India risk distribution).

5. **Risk Screen section -- the main table** (this is the part that's
   actually functional, not just styled like the reference): a sortable/
   filterable table rendering `results/universe_risk_screen.csv`, with
   these columns: **ticker symbol, full company name** (pull this from
   the ticker list CSVs -- S&P 500's `Security` column / Nifty 500's
   `Company Name` column, already available from `ticker_lists.py`),
   market (US/India), sector, **current/latest price**, **predicted
   next-day direction** (up/down arrow + the pooled model's confidence),
   risk category, risk score, volatility, VaR, Sharpe, max drawdown. Risk
   category as small color-coded pill badges (red-ish High, amber Medium,
   the palette's green Low) that still fit the black/white/lime aesthetic.
   Add a search/filter box (by symbol, company name, sector, market, risk
   category). **Every row is clickable** and opens that stock's detail
   view (see section 8 below). Each row also has a small pin/star icon to
   add it to the Watchlist without opening the detail view.

6. **CTA banner**: a light-gray rounded panel, bold headline + subtext on
   the left ("Want the full breakdown?"), a dark rounded CTA button, and a
   small illustration badge on the right, matching the reference's
   "Let's make things happen" section.

7. **Bottom stat/highlight row**: three dark rounded cards in a row (like
   the reference's "Case study" row), each showing one real, computed
   highlight instead of marketing copy -- e.g. "Highest risk stock this
   scan: X (risk score Y)", "Best pooled-model accuracy: X%", "Most
   volatile sector: X" -- each with a one-line explanation and a small
   "Learn more →" link scrolling to the relevant section.

## 8. Stock detail view

Clicking any stock (from the Risk Screen table, Watchlist, or search)
opens a detail view for that stock -- can be a distinct section/route in
the same page, doesn't need to be a separate HTML file. It should show
the *entire* analytics picture for that one stock, styled consistently
with the rest of the dashboard:

- Header: company name (large) + ticker + market, current price, and a
  prominent risk category badge.
- Price history chart (line chart of `close` over the fetched date range
  -- Chart.js or a simple canvas/SVG chart is fine, no need for a heavy
  charting library).
- All technical indicators from `features.py`, presented clearly: moving
  averages (SMA/EMA), RSI, MACD, Bollinger Bands -- either as small charts
  or as a clean stat grid with current values.
- All risk metrics from `risk_metrics.py`: annualized volatility,
  historical/parametric VaR, Sharpe ratio, max drawdown, beta -- as a stat
  grid, each with a one-line plain-language explanation (e.g. "VaR 95%:
  3.2% -- on the worst 5% of days, this stock has historically lost more
  than 3.2% in a day").
- Prediction section: the pooled model's next-day direction call,
  confidence/probability, and a short note on the model's overall
  accuracy (don't overstate confidence in a single prediction).
- A pin/unpin button to add or remove this stock from the Watchlist.
- A **"Compare"** button that activates split-screen mode (section 9).

## 9. Split-screen comparison

From a stock's detail view, clicking "Compare" opens a search/select
control to pick a second stock (any stock in the universe, not just the
watchlist). Once picked, the view splits into two side-by-side columns,
each showing that stock's detail view (header, price chart, indicators,
risk metrics, prediction) at a reduced width so both fit on screen. Keep
the two columns visually distinct but consistent with the palette (e.g. a
thin vertical divider). Include an easy way to swap out either side's
stock or exit comparison mode back to a single detail view.

**"Related stocks" suggestions in the search control**: before the user
types anything (and continuing to show above/alongside free-text search
results as they type), show a ranked list of stocks related to the one
they're already viewing -- don't make them think of a ticker to compare
against from scratch. Define "related" concretely, in this priority
order:

1. **Same sector/industry** as the primary stock (already have this from
   `ticker_lists.py` -- the S&P 500 CSV's sector column / Nifty 500 CSV's
   `Industry` column). This is the base filter -- pull the pool of
   candidates from here, across both markets (a same-sector US + India
   pairing, e.g. an IT services stock from each, is a genuinely
   interesting comparison, don't restrict to one market).
2. **Ranked within that pool by return correlation** -- compute the
   Pearson correlation of daily returns (`daily_return` from
   `features.py`, already available per ticker) between the primary stock
   and each same-sector candidate, over whatever history is cached. Show
   the top ~8, highest correlation first. This is a legitimate
   quantitative signal, not just a cosmetic sort -- highly correlated
   same-sector stocks are a genuinely meaningful comparison, and it's
   worth a line in the methodology page explaining that's how the
   suggestions are ordered.
3. If correlation can't be computed for some candidates (missing/short
   history), fall back to listing them alphabetically after the ranked
   ones rather than dropping them.

Each suggested stock in the list should show symbol, company name, sector
tag, and market badge, and be a single click to select as the comparison
stock. Cache the correlation computation per primary stock if it's slow to
recompute on every open (e.g. compute lazily and store in memory for the
session).

## 12. News sentiment analysis + personal portfolio integration

Two connected new features: a real sentiment score per stock (an actual
ML/NLP component, not just price-based metrics), and letting the user
bring in their own real holdings for personalized, portfolio-level
analysis.

### 12a. Sentiment analysis

- **Data source**: Alpha Vantage's `NEWS_SENTIMENT` endpoint (same free
  API key already wired up in `data_fetcher.py` -- add a new function
  there, e.g. `fetch_news_sentiment(symbol, api_key)`) returns recent news
  articles per ticker with Alpha Vantage's own sentiment score per
  article. Use this as the primary, always-available source.
- **The actual ML piece**: also run a proper pretrained financial
  sentiment model -- `ProsusAI/finbert` via the HuggingFace `transformers`
  library -- on the fetched headlines, to produce an independent sentiment
  score rather than relying solely on a third party's number. This is a
  legitimate, well-known model for exactly this task and is free to run
  locally (downloads once, ~400MB). Note in code comments that this adds
  a real dependency and one-time model download -- make it optional
  behind a flag if it's slow to set up, falling back to Alpha Vantage's
  built-in score alone if `transformers` isn't installed.
- Aggregate to one **sentiment score per stock** (e.g. -1 to +1, or
  Negative/Neutral/Positive), from the last ~20-50 recent articles, and
  show it in the stock detail view (section 8) as another stat, plus a
  small list of the actual recent headlines with their individual scores
  (transparency matters here -- don't just show a number with no
  evidence).
- Cache fetched news + computed scores per ticker (they don't need
  refreching every page load) similar to the price data caching pattern
  already used in `bulk_fetcher.py`.

### 12b. Portfolio input

Two ways in, both should work without any paid subscription:

1. **Manual add**: search the existing stock universe (reuse the same
   search/select control from the compare feature) and add a holding with
   quantity + average buy price. Store portfolio holdings in
   `localStorage` (same pattern as the Watchlist).
2. **CSV import**: let the user upload a holdings CSV exported from their
   broker (Zerodha Console, INDmoney, or any other -- these all export a
   similar shape: symbol, quantity, average price). Build a reasonably
   flexible parser that maps common column name variants (e.g. `Symbol`/
   `Instrument`/`Ticker`, `Qty`/`Quantity`, `Avg. cost`/`Average Price`/
   `Avg Price`) rather than requiring an exact format -- ask me for a
   sample export if you need to see the real column names for a specific
   broker.
3. **Optional/advanced -- Zerodha Kite Connect live sync**: only build
   this if the above two are solid first. It requires *my own* paid Kite
   Connect API subscription and a daily login flow (access tokens expire
   every day, no persistent auto-login) -- so implement it as a clearly
   separate, clearly-labeled "Advanced: connect Zerodha" option that
   expects the user to provide their own API key/secret and complete the
   daily login redirect, not something that "just works" out of the box.
   Don't block anything else on this working.

INDmoney has no public API -- CSV import is the only integration path for
it, there's no live-sync option to build.

### 12c. Portfolio analysis page

A new "Portfolio" nav section (alongside Overview / Risk Screen /
Watchlist / Methodology) showing:

- **Holdings list**: reuse the stock detail card layout from section 11
  for each holding, but in the reference screenshot's portfolio framing --
  invested amount, current value, P&L (₹/$ and %), quantity, average
  price -- computed from the holding's entered qty/avg price and the
  latest fetched close price.
- **Portfolio-level risk** (this needs to be done correctly, not naively):
  portfolio volatility is *not* just the weighted average of each
  holding's individual volatility -- it depends on the covariance between
  holdings (diversification reduces risk when holdings aren't perfectly
  correlated). Compute portfolio variance properly:
  `portfolio_variance = w^T * Cov * w` where `w` is the vector of
  position weights (by current value) and `Cov` is the covariance matrix
  of daily returns across the holdings (reuse the correlation/covariance
  computation approach from section 9's related-stocks feature). Report
  portfolio annualized volatility, portfolio VaR, and portfolio Sharpe
  ratio from this, not a naive average.
- **Concentration/diversification**: sector breakdown of the portfolio by
  value (e.g. a simple bar or donut chart), and a flag if any single
  holding or sector is over some concentration threshold (e.g. >25% of
  portfolio value) -- explain in one line why concentration itself is a
  risk factor.
- **Aggregate sentiment**: value-weighted average of each holding's
  sentiment score from 12a, so the portfolio has one overall sentiment
  reading alongside per-holding detail.
- **Individual risk flags**: call out any holding that's individually
  categorized High risk in the main risk screen, since that's actionable
  even before looking at portfolio-level numbers.

Build order: 12b (manual add first, CSV import second, Kite Connect last
and optional) -> 12c using whatever portfolio is entered -> 12a sentiment,
wired into both the stock detail view and the portfolio page. Show me
manual add + the portfolio risk page working on 2-3 holdings before adding
CSV import or sentiment.

## 10. Watchlist

A dedicated "Watchlist" nav section listing only the stocks the user has
pinned (from the Risk Screen table's pin icon or a detail view's pin
button). Same table format as the Risk Screen (symbol, company name,
price, prediction, risk category, etc.) but only pinned rows, plus an
unpin control on each row. If the watchlist is empty, show a friendly
empty state pointing back to the Risk Screen.

**Persistence**: since this is a plain local HTML/JS dashboard running in
the user's own browser (not a hosted app), use `localStorage` to persist
the pinned-symbol list across page reloads -- that's the correct and
expected approach here (this restriction only applies to Claude.ai
artifacts, not to a standalone project you're building for the user's own
machine).

**General styling notes**: generously rounded corners (~20px) on cards and
buttons, thin borders on white cards, subtle drop shadows, plenty of
white/black negative space -- don't cram the layout. Keep it responsive
enough to not break on a laptop screen; a polished desktop view matters
more than mobile for this project.

Wire this dashboard to read the actual CSV outputs (don't hardcode sample
numbers into the HTML) -- either via a small local script that injects the
data at build time, or client-side JS that fetches the CSV, whichever is
simpler to keep working as the underlying data refreshes. Per-stock detail
data (for section 8) needs the individual price/feature history, not just
the summary row -- read from the cached per-ticker files in
`data/universe/<ticker>.csv` plus `features.py`'s output, computed
client-side or pre-computed and cached per ticker, whichever is faster to
build and doesn't make the page load slowly across ~1,000 stocks.

Build the dashboard incrementally: get the Risk Screen table working and
clickable first, then the detail view, then compare mode, then the
watchlist -- show me each stage working before moving to the next rather
than building all of it blind.

## 11. Stock detail card redesign + real charting (replaces the current chart)

I have a reference screenshot (a real trading app's holding detail screen)
whose *layout pattern* I want the stock detail view (section 8) to follow
-- not its exact color scheme (keep our existing black/white/lime
palette), its information architecture:

- **Header block**: stock name large and bold, exchange/market badge next
  to it (NSE/US), current price large with the day's change in
  green (up) or red (down) -- red/green for price movement is the one
  place it's fine to deviate from pure lime-only, since it's a universal
  finance convention and matches what users expect.
- **Primary action row**: two side-by-side pill buttons, full-width. Since
  this is an analytics/screening tool, not a trading app, adapt the
  actions to what we actually do -- e.g. "Pin to Watchlist" (or "Unpin" if
  already pinned) and "Compare" (opens split-screen mode from section 9),
  styled like the reference's ADD/EXIT button row (solid rounded
  rectangles, one primary-colored, one secondary).
- **Secondary link row**: small icon + text links, like the reference's
  "View chart" / "Option chain" row -- ours would be "Full chart"
  (expands the chart, see below) and "Methodology" (jumps to how this
  stock's risk score was computed).
- **Stat block**: clean label/value rows matching the reference's
  Investment/Qty/Avg price layout, but with our actual metrics --
  annualized volatility, VaR (95%), Sharpe ratio, max drawdown, risk
  category, predicted next-day direction + confidence. Same visual
  rhythm: label left, value right, alternating subtle row shading.
- **Chart preview**: a compact sparkline-style preview of the price trend
  at the bottom of the card (like the reference's "Trend 52 weeks"
  chart), which expands into the full interactive chart (below) when
  clicked or via the "Full chart" link.

**The chart itself needs to be replaced, not just restyled** -- the
current one is low-quality and has no zoom, pan, or point-level price
readout, which makes it useless for real analysis. Rebuild it using a
proper financial charting library instead of a basic canvas/SVG line:

- Use **TradingView's Lightweight Charts library**
  (`lightweight-charts`, free, open-source, loaded via CDN or npm --
  this is the same category of library real trading platforms are built
  on, including the one in my reference screenshot) for the full chart
  view. It gives you, out of the box: smooth pan (click-drag), zoom
  (scroll wheel / pinch), and a crosshair that shows the exact date and
  price at whatever point the cursor is over -- all three of which are
  what's currently missing.
- If Lightweight Charts turns out to be awkward to wire up, the fallback
  is Chart.js plus the `chartjs-plugin-zoom` (pan/zoom) and a crosshair/
  tooltip plugin -- but try Lightweight Charts first, it's purpose-built
  for exactly this.
- Add simple range toggle buttons above the chart -- 1W / 1M / 3M / 1Y /
  Max -- matching the reference's "Trend 52 weeks" label implying a
  selectable range, using whatever history is available in
  `data/universe/<ticker>.csv`.
- The compact preview sparkline (in the card) can stay a simple lightweight
  render (no need for full interactivity there) -- only the expanded
  "Full chart" view needs the upgraded library.

Show me the redesigned card and the new chart working on one single stock
before rolling it out across the full table.

---
