"""
sentiment.py

News sentiment per stock -- the ML/NLP piece, not just price-based metrics.

Primary source: Alpha Vantage's NEWS_SENTIMENT endpoint (data_fetcher.py's
fetch_news_sentiment), which returns recent articles per ticker with Alpha
Vantage's own per-article sentiment score attached. Always available with
just the free API key already used elsewhere in this project.

Optional second source: a real pretrained financial sentiment model --
ProsusAI/finbert via HuggingFace `transformers` -- run locally on the same
headlines, for an independent score rather than relying solely on a third
party's number. This is gated behind --finbert because it downloads a
~400MB model on first use; without the flag, or if `transformers` isn't
installed, this falls back cleanly to the Alpha Vantage score alone.

Aggregates to one sentiment score per stock (-1..+1, plus Alpha Vantage's
own Bearish..Bullish label bands) from the ~20-50 most recent articles, and
keeps the individual headlines with their per-article scores for
transparency -- the dashboard shows both, not just a bare number.

Caches to data/sentiment/<ticker>.json, same pattern as bulk_fetcher.py's
price cache, so news isn't refetched every page load.

NOTE on testing: Alpha Vantage's free-tier 'demo' API key does not return
real data for NEWS_SENTIMENT for any symbol (unlike TIME_SERIES_DAILY,
whose demo key works for IBM) -- so unlike the rest of this project, this
endpoint's live response could not be captured or validated during
development. The parsing/aggregation logic below is instead validated
against a hand-built, realistic sample response that matches Alpha
Vantage's documented schema (data/sample/AAPL_news_sentiment_sample.json,
--use_sample). It should work normally against the real endpoint with your
own free key -- https://www.alphavantage.co/support/#api-key.

Usage:
    # offline pipeline test, no API key needed
    python -m src.sentiment --symbols AAPL --use_sample

    # live, Alpha Vantage score only
    python -m src.sentiment --symbols AAPL MSFT --api_key YOUR_KEY

    # live, plus FinBERT (downloads ~400MB the first time)
    python -m src.sentiment --symbols AAPL --api_key YOUR_KEY --finbert
"""

import argparse
import json
import os
import time

from src.data_fetcher import fetch_news_sentiment

SENTIMENT_DIR = "data/sentiment"
SAMPLE_PATH = "data/sample/AAPL_news_sentiment_sample.json"


def label_from_score(score: float) -> str:
    """Alpha Vantage's own threshold bands, applied to our aggregate score
    too so the label is consistent whether it came from one article or
    the aggregate of fifty."""
    if score <= -0.35:
        return "Bearish"
    if score <= -0.15:
        return "Somewhat-Bearish"
    if score < 0.15:
        return "Neutral"
    if score < 0.35:
        return "Somewhat-Bullish"
    return "Bullish"


def _ticker_score(article: dict, symbol: str) -> float:
    """Prefer the article's ticker-specific sentiment score over its overall
    score -- an article mostly about a different company that mentions ours
    in passing shouldn't move our score by its overall tone."""
    for ts in article.get("ticker_sentiment", []):
        if ts.get("ticker", "").upper() == symbol.upper():
            try:
                return float(ts["ticker_sentiment_score"])
            except (KeyError, ValueError):
                break
    try:
        return float(article["overall_sentiment_score"])
    except (KeyError, ValueError):
        return 0.0


def score_with_finbert(headlines: list):
    """Returns a list of scores in [-1, 1] aligned with `headlines`, or None
    if `transformers` isn't installed -- the caller falls back to the
    Alpha Vantage score alone in that case, per the project's honesty
    pattern of not silently failing or overclaiming a dependency that
    isn't there."""
    try:
        from transformers import pipeline
    except ImportError:
        return None

    classifier = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    results = classifier(headlines, truncation=True)
    scored = []
    for r in results:
        label = r["label"].lower()
        if label == "positive":
            scored.append(r["score"])
        elif label == "negative":
            scored.append(-r["score"])
        else:
            scored.append(0.0)
    return scored


def aggregate_sentiment(feed_data: dict, symbol: str, use_finbert: bool = False) -> dict:
    articles = feed_data.get("feed", [])
    if not articles:
        return {
            "symbol": symbol, "score": None, "label": None, "n_articles": 0,
            "articles": [], "finbert_available": False,
        }

    scored_articles = []
    for a in articles:
        scored_articles.append({
            "title": a.get("title", ""),
            "url": a.get("url", ""),
            "source": a.get("source", ""),
            "time_published": a.get("time_published", ""),
            "score": round(_ticker_score(a, symbol), 4),
        })

    avg_score = sum(a["score"] for a in scored_articles) / len(scored_articles)

    finbert_scores = None
    if use_finbert:
        finbert_scores = score_with_finbert([a["title"] for a in scored_articles])
        if finbert_scores is not None:
            for a, fs in zip(scored_articles, finbert_scores):
                a["finbert_score"] = round(fs, 4)

    scored_articles.sort(key=lambda a: a["time_published"], reverse=True)

    payload = {
        "symbol": symbol,
        "score": round(avg_score, 4),
        "label": label_from_score(avg_score),
        "n_articles": len(scored_articles),
        "articles": scored_articles[:20],
        "finbert_available": finbert_scores is not None,
    }
    if finbert_scores is not None:
        payload["finbert_score"] = round(sum(finbert_scores)/len(finbert_scores), 4)
        payload["finbert_label"] = label_from_score(payload["finbert_score"])
    return payload


def export_ticker_sentiment(symbol: str, api_key: str = None, out_dir: str = SENTIMENT_DIR,
                             use_finbert: bool = False, use_sample: bool = False):
    if use_sample:
        with open(SAMPLE_PATH) as f:
            feed_data = json.load(f)
    else:
        feed_data = fetch_news_sentiment(symbol, api_key)

    payload = aggregate_sentiment(feed_data, symbol, use_finbert=use_finbert)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{symbol}.json"), "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=["AAPL"])
    parser.add_argument("--api_key", type=str, default=os.environ.get("ALPHAVANTAGE_API_KEY"))
    parser.add_argument("--out_dir", type=str, default=SENTIMENT_DIR)
    parser.add_argument("--finbert", action="store_true",
                         help="also score headlines with ProsusAI/finbert (downloads ~400MB on first use)")
    parser.add_argument("--use_sample", action="store_true",
                         help="use the bundled sample response instead of calling the live API (no key needed)")
    args = parser.parse_args()

    if not args.use_sample and not args.api_key:
        raise SystemExit("No API key found (set ALPHAVANTAGE_API_KEY or pass --api_key), "
                          "or pass --use_sample to test offline with the bundled sample response.")

    for i, symbol in enumerate(args.symbols):
        print(f"Fetching news sentiment for {symbol} ({i+1}/{len(args.symbols)})...")
        payload = export_ticker_sentiment(
            symbol, api_key=args.api_key, out_dir=args.out_dir,
            use_finbert=args.finbert, use_sample=args.use_sample,
        )
        print(f"  {symbol}: score={payload['score']} label={payload['label']} "
              f"({payload['n_articles']} articles"
              f"{', finbert=' + str(payload.get('finbert_score')) if payload['finbert_available'] else ''})")
        if not args.use_sample and i < len(args.symbols) - 1:
            time.sleep(13)  # stay under the free tier's 5 req/min, same pacing as data_fetcher.py

    print(f"\nSaved to {args.out_dir}/")
