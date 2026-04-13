"""
Data Module: Fetch Market Data
==============================
Pulls live and historical market data from Yahoo Finance and computes
the feature set required by the Crisis Detector model.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Optional

from utils.config import (
    NIFTY50_TICKER,
    SENSEX_TICKER,
    INDIA_VIX_TICKER,
    BANKNIFTY_TICKER,
    SP500_TICKER,
    GOLD_TICKER,
    USDINR_TICKER,
    CRUDE_TICKER,
    LOOKBACK_252D,
    VOL_WINDOW,
    CACHE_TTL_SECONDS,
)
from utils.helpers import get_logger, safe_pct_change, compute_rolling_volatility, retry

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Individual Ticker Fetching
# ---------------------------------------------------------------------------

@retry(max_attempts=3, delay_seconds=2.0)
def fetch_ticker_history(ticker: str, period: str = "1y") -> pd.Series:
    """
    Download adjusted close price history for a single ticker.

    Args:
        ticker: Yahoo Finance ticker symbol.
        period: Data period (e.g., "1y", "6mo", "3mo").

    Returns:
        Pandas Series of closing prices indexed by date.
    """
    logger.debug(f"Fetching {ticker}…")
    data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError(f"No data returned for ticker: {ticker}")
    return data["Close"].dropna()


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def compute_features_for_series(prices: pd.Series, prefix: str) -> pd.DataFrame:
    """
    From a price series, compute standard return and volatility features.

    Returns a DataFrame with columns prefixed by `prefix`.
    """
    returns = safe_pct_change(prices, 1)
    features = pd.DataFrame(index=prices.index)
    features[f"{prefix}_ret_1d"]  = returns
    features[f"{prefix}_ret_5d"]  = safe_pct_change(prices, 5)
    features[f"{prefix}_ret_20d"] = safe_pct_change(prices, 20)
    features[f"{prefix}_vol_20d"] = compute_rolling_volatility(returns, VOL_WINDOW)
    return features


def fetch_all_features(period: str = "1y") -> pd.DataFrame:
    """
    Assemble the full feature matrix for the panic detection model.
    Aligns all series to a common date index.

    Returns:
        DataFrame with one row per trading day; latest row = today.
    """
    logger.info("Fetching all market features…")

    # --- Core Indian Market ---
    nifty_prices   = fetch_ticker_history(NIFTY50_TICKER, period)
    vix_prices     = fetch_ticker_history(INDIA_VIX_TICKER, period)
    banknifty      = fetch_ticker_history(BANKNIFTY_TICKER, period)

    # --- Global Proxies ---
    sp500_prices   = fetch_ticker_history(SP500_TICKER, period)
    gold_prices    = fetch_ticker_history(GOLD_TICKER, period)
    usdinr_prices  = fetch_ticker_history(USDINR_TICKER, period)
    crude_prices   = fetch_ticker_history(CRUDE_TICKER, period)

    # --- Feature Computation ---
    nifty_feats    = compute_features_for_series(nifty_prices, "nifty")
    sp500_feats    = compute_features_for_series(sp500_prices, "sp500")
    banknifty_feats = compute_features_for_series(banknifty, "banknifty")

    # Align all to Nifty index (Indian trading days)
    features = nifty_feats.copy()
    features = features.join(sp500_feats[["sp500_ret_1d"]], how="left")
    features = features.join(banknifty_feats[["banknifty_ret_1d"]], how="left")

    # VIX level and change
    vix_aligned = vix_prices.reindex(features.index, method="ffill")
    features["vix_level"]     = vix_aligned
    features["vix_change_1d"] = safe_pct_change(vix_aligned, 1)

    # Single-day returns for global proxies
    features["gold_ret_1d"]    = safe_pct_change(
        gold_prices.reindex(features.index, method="ffill"), 1
    )
    features["usdinr_change_1d"] = safe_pct_change(
        usdinr_prices.reindex(features.index, method="ffill"), 1
    )
    features["crude_ret_1d"]   = safe_pct_change(
        crude_prices.reindex(features.index, method="ffill"), 1
    )

    # Placeholder columns — to be integrated from NSE breadth data in v2
    features["advance_decline_ratio"] = np.nan
    features["pct_stocks_below_200dma"] = np.nan
    features["fii_net_flow_cr"] = np.nan
    features["dii_net_flow_cr"] = np.nan

    features = features.ffill().bfill()
    logger.info(f"Feature matrix ready: {features.shape[0]} rows × {features.shape[1]} cols")
    return features


# ---------------------------------------------------------------------------
# Quick Snapshot (for Dashboard)
# ---------------------------------------------------------------------------

def get_market_snapshot() -> dict:
    """
    Fetch current-day market snapshot for dashboard display.

    Returns:
        Dict with key index levels, changes, and VIX.
    """
    snapshot = {}
    tickers = {
        "Nifty 50":  NIFTY50_TICKER,
        "Sensex":    SENSEX_TICKER,
        "India VIX": INDIA_VIX_TICKER,
        "Bank Nifty": BANKNIFTY_TICKER,
    }
    for name, ticker in tickers.items():
        try:
            data = yf.Ticker(ticker).fast_info
            snapshot[name] = {
                "last_price": round(data.get("last_price", 0), 2),
                "day_change_pct": round(
                    (data.get("last_price", 0) - data.get("previous_close", 1))
                    / data.get("previous_close", 1) * 100, 2
                ),
            }
        except Exception as e:
            logger.warning(f"Could not fetch snapshot for {name}: {e}")
            snapshot[name] = {"last_price": None, "day_change_pct": None}
    return snapshot
