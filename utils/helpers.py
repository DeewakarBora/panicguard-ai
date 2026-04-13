"""
PanicGuard AI — Shared Utility Functions
General-purpose helpers used across agents, models, and the dashboard.
"""

import logging
import time
from datetime import datetime, timedelta
from functools import wraps
from typing import Optional

import pandas as pd

from utils.config import (
    LOG_LEVEL,
    PANIC_LABELS,
    PANIC_COLORS,
    PANIC_THRESHOLD_LOW,
    PANIC_THRESHOLD_MEDIUM,
    PANIC_THRESHOLD_HIGH,
)

# ============================================================
# Logging Setup
# ============================================================

def get_logger(name: str) -> logging.Logger:
    """Return a consistently formatted logger for any module."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    return logger


# ============================================================
# Panic Score Utilities
# ============================================================

def classify_panic_score(score: float) -> dict:
    """
    Map a raw panic score (0–1) to a human-readable regime label and color.

    Args:
        score: Float between 0 and 1.

    Returns:
        dict with keys: 'label', 'color', 'emoji', 'description'
    """
    if score < PANIC_THRESHOLD_LOW:
        return {
            "label": "Normal Market",
            "color": PANIC_COLORS["normal"],
            "emoji": "✅",
            "description": "Markets are behaving within normal parameters. No intervention needed.",
        }
    elif score < PANIC_THRESHOLD_MEDIUM:
        return {
            "label": "Elevated Stress",
            "color": PANIC_COLORS["elevated"],
            "emoji": "⚠️",
            "description": "Markets show signs of stress. Monitor closely but avoid impulsive decisions.",
        }
    elif score < PANIC_THRESHOLD_HIGH:
        return {
            "label": "Crisis Mode",
            "color": PANIC_COLORS["crisis"],
            "emoji": "🔴",
            "description": "Significant market distress detected. High risk of panic decisions. Behavioral coach activated.",
        }
    else:
        return {
            "label": "Extreme Panic",
            "color": PANIC_COLORS["extreme"],
            "emoji": "🚨",
            "description": "Extreme fear in the market. This is historically when long-term investors are made — do NOT sell.",
        }


# ============================================================
# Date & Time Utilities
# ============================================================

def get_trading_days_back(n: int) -> str:
    """Return a date string N trading days ago (approximate, ignores holidays)."""
    today = datetime.today()
    days_back = int(n * 1.4)  # buffer for weekends
    return (today - timedelta(days=days_back)).strftime("%Y-%m-%d")


def is_market_open() -> bool:
    """
    Rough check: is it currently Indian market hours?
    NSE trades Mon–Fri, 09:15–15:30 IST.
    """
    now = datetime.utcnow() + timedelta(hours=5, minutes=30)  # IST
    if now.weekday() >= 5:  # Saturday or Sunday
        return False
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close


# ============================================================
# DataFrame Utilities
# ============================================================

def safe_pct_change(series: pd.Series, periods: int = 1) -> pd.Series:
    """Compute percentage change, handling edge cases gracefully."""
    return series.pct_change(periods=periods).fillna(0)


def compute_rolling_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """Annualized rolling volatility from daily returns."""
    return returns.rolling(window=window).std() * (252 ** 0.5)


# ============================================================
# Portfolio Utilities
# ============================================================

def calculate_drawdown(prices: pd.Series) -> pd.Series:
    """Compute the drawdown series from a price series."""
    rolling_max = prices.cummax()
    drawdown = (prices - rolling_max) / rolling_max
    return drawdown


def format_inr(value: float) -> str:
    """Format a number as Indian Rupees with lakh/crore suffix."""
    if abs(value) >= 1e7:
        return f"₹{value / 1e7:.2f} Cr"
    elif abs(value) >= 1e5:
        return f"₹{value / 1e5:.2f} L"
    else:
        return f"₹{value:,.0f}"


# ============================================================
# Retry Decorator
# ============================================================

def retry(max_attempts: int = 3, delay_seconds: float = 2.0, exceptions=(Exception,)):
    """
    Decorator that retries a function on failure.

    Usage:
        @retry(max_attempts=3, delay_seconds=1.0, exceptions=(ConnectionError,))
        def fetch_data(): ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger("retry")
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        logger.error(f"[{func.__name__}] Failed after {max_attempts} attempts: {e}")
                        raise
                    logger.warning(f"[{func.__name__}] Attempt {attempt} failed: {e}. Retrying in {delay_seconds}s…")
                    time.sleep(delay_seconds)
        return wrapper
    return decorator
