"""
PanicGuard AI — Central Configuration
All constants, thresholds, and environment-variable bindings live here.
Import this module wherever configuration is needed.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file (does nothing in production where env vars are set directly)
load_dotenv()

# Project root — resolved relative to this file so paths work regardless of CWD
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ============================================================
# App Meta
# ============================================================
APP_NAME = "PanicGuard AI"
APP_VERSION = "1.0.0"
APP_ENV = os.getenv("APP_ENV", "development")  # "development" | "production"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ============================================================
# LLM Configuration
# ============================================================
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")  # "anthropic" | "openai"

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# LLM generation parameters
LLM_MAX_TOKENS = 1024
LLM_TEMPERATURE = 0.4          # Lower = more factual, less creative

# ============================================================
# Panic Detection Thresholds
# ============================================================
PANIC_THRESHOLD_LOW = float(os.getenv("PANIC_THRESHOLD_LOW", 0.30))
PANIC_THRESHOLD_MEDIUM = float(os.getenv("PANIC_THRESHOLD_MEDIUM", 0.60))
PANIC_THRESHOLD_HIGH = float(os.getenv("PANIC_THRESHOLD_HIGH", 0.80))

PANIC_LABELS = {
    "normal":   (0.00, PANIC_THRESHOLD_LOW),
    "elevated": (PANIC_THRESHOLD_LOW, PANIC_THRESHOLD_MEDIUM),
    "crisis":   (PANIC_THRESHOLD_MEDIUM, PANIC_THRESHOLD_HIGH),
    "extreme":  (PANIC_THRESHOLD_HIGH, 1.00),
}

PANIC_COLORS = {
    "normal":   "#22c55e",   # green
    "elevated": "#f59e0b",   # amber
    "crisis":   "#ef4444",   # red
    "extreme":  "#7c3aed",   # violet — maximum alarm
}

# ============================================================
# Market Data Configuration
# ============================================================
# Primary Indian market indices (Yahoo Finance tickers)
NIFTY50_TICKER = "^NSEI"
SENSEX_TICKER = "^BSESN"
INDIA_VIX_TICKER = "^INDIAVIX"
BANKNIFTY_TICKER = "^NSEBANK"

# Global proxies
SP500_TICKER = "^GSPC"
GOLD_TICKER = "GC=F"
USDINR_TICKER = "USDINR=X"
CRUDE_TICKER = "CL=F"

# Default lookback windows (in trading days)
LOOKBACK_1D = 1
LOOKBACK_5D = 5
LOOKBACK_20D = 20
LOOKBACK_60D = 60
LOOKBACK_252D = 252            # ~1 trading year

# Volatility rolling window
VOL_WINDOW = 20

# Data cache TTL in seconds
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", 300))

# ============================================================
# ML Model Configuration
# ============================================================
MODEL_SAVE_PATH = str(_PROJECT_ROOT / "models" / "saved_models" / "panic_detector_xgb.joblib")
SCALER_SAVE_PATH = str(_PROJECT_ROOT / "models" / "saved_models" / "feature_scaler.joblib")

# XGBoost hyperparameters (baseline — tune via cross-validation)
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "use_label_encoder": False,
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
}

# Feature list (must match training order)
FEATURE_COLUMNS = [
    "nifty_ret_1d",
    "nifty_ret_5d",
    "nifty_ret_20d",
    "nifty_vol_20d",
    "vix_level",
    "vix_change_1d",
    "sp500_ret_1d",
    "usdinr_change_1d",
    "gold_ret_1d",
    "crude_ret_1d",
    "advance_decline_ratio",
    "pct_stocks_below_200dma",
    "fii_net_flow_cr",
    "dii_net_flow_cr",
    "banknifty_ret_1d",
]

# ============================================================
# Portfolio Analyzer Configuration
# ============================================================
# Minimum portfolio size for meaningful analysis
MIN_PORTFOLIO_VALUE = 10_000   # INR

# Historical crash reference windows (start, end date strings)
CRASH_REFERENCES = {
    "Global Financial Crisis (2008)": ("2008-01-01", "2009-03-31"),
    "Euro Debt Crisis (2011)":        ("2011-07-01", "2011-12-31"),
    "China Slowdown (2015)":          ("2015-08-01", "2016-02-29"),
    "IL&FS Crisis (2018)":            ("2018-09-01", "2019-03-31"),
    "COVID Crash (2020)":             ("2020-02-01", "2020-04-30"),
    "Rate Hike Selloff (2022)":       ("2022-01-01", "2022-06-30"),
    "April 2026 Correction":          ("2026-04-01", "2026-04-30"),
}

# ============================================================
# Behavioral Coach — System Prompt
# ============================================================
COACH_SYSTEM_PROMPT = """
You are PanicGuard AI's Behavioral Coach — a calm, empathetic financial educator
specializing in behavioral finance. Your role is to help retail investors in India
make rational, evidence-based decisions during periods of market stress.

RULES:
1. You NEVER give specific investment advice, stock tips, or buy/sell recommendations.
2. You ALWAYS acknowledge the investor's emotions before providing data or context.
3. You use behavioral finance frameworks: loss aversion, recency bias, herd behavior,
   anchoring, and availability heuristic.
4. You ground every response in historical data and base rates.
5. You are NOT SEBI-registered. Always clarify this if asked about financial advice.
6. Keep responses concise (3–5 sentences max per turn unless the user asks for more).
7. Always end with a question to keep the investor engaged and reflecting.
"""

# ============================================================
# Dashboard Configuration
# ============================================================
DASHBOARD_TITLE = "🛡️ PanicGuard AI"
DASHBOARD_SUBTITLE = "Your behavioral firewall against panic-driven investment decisions."
REFRESH_INTERVAL_SEC = 300     # Auto-refresh interval for live data (5 minutes)

# Number of top SHAP features to display in the UI
SHAP_TOP_K_FEATURES = 5
