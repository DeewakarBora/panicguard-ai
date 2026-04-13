"""
PanicGuard AI — Complete ML Training Pipeline
==============================================
Trains an XGBoost classifier to detect market panic regimes in Indian equity
markets.  The model powers the Crisis Detector agent in preparation for
real-time deployment on Streamlit Cloud.

Full pipeline
    1. Fetch 10 years of Nifty 50 + auxiliary data via yfinance
    2. Engineer 15 technical / macro features
    3. Label panic windows using a composite rule (drawdown + volatility + streak)
    4. Train XGBoost with walk-forward time-series CV and SMOTE
    5. Generate SHAP explanations & waterfall plots
    6. Expose a ``predict_panic_score()`` function for inference
    7. Backtest against COVID-2020, Russia-Ukraine-2022, April-2026 crashes

Run standalone:
    python -m models.train_panic_model          # from repo root
    python models/train_panic_model.py          # also works with sys.path hack below
"""

from __future__ import annotations

import os
import sys
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")                    # headless — must come before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so ``python models/train_panic_model.py``
# works even when launched from the models/ directory.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# --- Project imports (existing modules) ------------------------------------
from utils.config import (
    CRASH_REFERENCES,
    CRUDE_TICKER,
    INDIA_VIX_TICKER,
    MODEL_SAVE_PATH,
    NIFTY50_TICKER,
    SCALER_SAVE_PATH,
    SHAP_TOP_K_FEATURES,
    VOL_WINDOW,
    XGB_PARAMS,
)
from utils.helpers import (
    calculate_drawdown,
    compute_rolling_volatility,
    get_logger,
    retry,
    safe_pct_change,
)

logger = get_logger("models.train_panic_model")

# ============================================================
# Constants (pipeline-specific)
# ============================================================

# Output directories — all artefacts land inside models/
MODELS_DIR          = Path(__file__).resolve().parent
SAVED_MODELS_DIR    = MODELS_DIR / "saved_models"
PLOTS_DIR           = MODELS_DIR / "plots"
METRICS_DIR         = MODELS_DIR / "metrics"

# Feature column names produced by this pipeline.
# This is the *canonical* list — config.FEATURE_COLUMNS is kept in sync at
# save time by writing a feature_columns.json next to the model.
TRAINING_FEATURES: list[str] = [
    "daily_return",
    "rolling_volatility_20d",
    "rolling_volatility_50d",
    "drawdown_from_peak",
    "rsi_14",
    "macd_signal",
    "vix_level",
    "crude_oil_change",
    "consecutive_red_days",
    "distance_from_200dma",
    "weekly_return",
    "monthly_return",
    "fii_flow_proxy",
    "bollinger_band_position",
    "crash_severity_score",
]

# Walk-forward CV settings
CV_TRAIN_YEARS = 4    # minimum years of training data per fold
CV_TEST_MONTHS = 6    # size of each test fold
CV_STEP_MONTHS = 3    # step between folds

# Panic labelling thresholds
DRAWDOWN_THRESHOLD   = -0.10      # –10 % from 52-week high
VOL_MULTIPLE_THRESH  = 2.0        # 2× the rolling-mean volatility
RED_DAYS_THRESHOLD   = 3          # ≥ 3 consecutive down days

# Backtest windows (start, peak_panic, end, label)
BACKTEST_EVENTS = [
    {
        "name":        "COVID-19 Crash (2020)",
        "start":       "2020-01-01",
        "peak_panic":  "2020-03-23",
        "end":         "2020-09-30",
        "hold_gain":   112.0,    # % gain from bottom to 18 months later
        "sell_loss":   -15.0,    # typical realised loss if sold at bottom
    },
    {
        "name":        "Russia-Ukraine Selloff (2022)",
        "start":       "2022-01-01",
        "peak_panic":  "2022-06-17",
        "end":         "2022-12-31",
        "hold_gain":   28.0,
        "sell_loss":   -10.0,
    },
    {
        "name":        "April 2026 Tariff Crash",
        "start":       "2026-03-01",
        "peak_panic":  "2026-04-10",
        "end":         "2026-04-14",      # ongoing — use today
        "hold_gain":   None,              # TBD
        "sell_loss":   -12.0,
    },
]


# ============================================================
# §1  DATA FETCHING
# ============================================================

@retry(max_attempts=3, delay_seconds=3.0, exceptions=(Exception,))
def _download_yf(ticker: str, period: str = "10y") -> pd.DataFrame:
    """Download OHLCV data for *ticker* via yfinance with retry."""
    import yfinance as yf
    logger.info(f"  ↳ downloading {ticker} ({period})…")
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"Empty DataFrame for {ticker}")
    # yfinance ≥ 0.2.38 may return MultiIndex columns for single tickers
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def fetch_training_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch Nifty-50, India VIX, and Crude Oil data for 10 years.

    Returns
    -------
    nifty_df : pd.DataFrame   — OHLCV for ^NSEI
    aux_df   : pd.DataFrame   — VIX level + crude close, aligned to nifty dates
    """
    logger.info("=" * 60)
    logger.info("STEP 1 / 7 — Fetching 10-year market data")
    logger.info("=" * 60)

    nifty_df = _download_yf(NIFTY50_TICKER, period="10y")
    logger.info(f"  Nifty 50: {len(nifty_df)} rows  ({nifty_df.index.min().date()} → {nifty_df.index.max().date()})")

    # --- India VIX (graceful fallback) ---------------------------------
    try:
        vix_df = _download_yf(INDIA_VIX_TICKER, period="10y")
        vix_series = vix_df["Close"].rename("vix_level")
        logger.info(f"  India VIX: {len(vix_df)} rows")
    except Exception as e:
        logger.warning(f"  India VIX fetch failed ({e}). Generating synthetic VIX from Nifty volatility.")
        vix_series = None   # will be synthesised downstream

    # --- Crude Oil -----------------------------------------------------
    try:
        crude_df = _download_yf(CRUDE_TICKER, period="10y")
        crude_series = crude_df["Close"].rename("crude_close")
        logger.info(f"  Crude Oil: {len(crude_df)} rows")
    except Exception as e:
        logger.warning(f"  Crude Oil fetch failed ({e}). Feature will be zero-filled.")
        crude_series = pd.Series(dtype=float, name="crude_close")

    # --- Align to Nifty trading calendar --------------------------------
    aux = pd.DataFrame(index=nifty_df.index)
    if vix_series is not None:
        aux["vix_level"] = vix_series.reindex(aux.index, method="ffill")
    if not crude_series.empty:
        aux["crude_close"] = crude_series.reindex(aux.index, method="ffill")

    aux = aux.ffill().bfill()    # fill any remaining NaNs at edges
    return nifty_df, aux


def _generate_demo_data(n_days: int = 2500) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Produce deterministic synthetic Nifty data so the pipeline can run
    end-to-end even when yfinance is unreachable (demo-mode requirement).
    """
    logger.warning("⚠ Using synthetic demo data — model will NOT reflect real markets")
    rng = np.random.RandomState(42)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days, freq="B")

    # Random walk with drift + injected crash windows
    log_returns = rng.normal(0.0004, 0.012, size=n_days)
    # Inject a ~35 % crash around index 2000 (≈ position of 2020)
    log_returns[1900:1930] = rng.normal(-0.025, 0.025, size=30)
    # Inject a small selloff later
    log_returns[2200:2215] = rng.normal(-0.015, 0.015, size=15)

    prices = 10_000 * np.exp(np.cumsum(log_returns))

    nifty_df = pd.DataFrame(
        {
            "Open":   prices * (1 + rng.normal(0, 0.003, n_days)),
            "High":   prices * (1 + np.abs(rng.normal(0, 0.005, n_days))),
            "Low":    prices * (1 - np.abs(rng.normal(0, 0.005, n_days))),
            "Close":  prices,
            "Volume": rng.randint(50_000_000, 300_000_000, size=n_days).astype(float),
        },
        index=dates,
    )
    aux_df = pd.DataFrame(
        {
            "vix_level":    15 + 10 * compute_rolling_volatility(pd.Series(log_returns, index=dates), 20).fillna(0.15),
            "crude_close":  60 + 20 * np.sin(np.linspace(0, 8 * np.pi, n_days)) + rng.normal(0, 2, n_days),
        },
        index=dates,
    )
    return nifty_df, aux_df


# ============================================================
# §2  FEATURE ENGINEERING
# ============================================================

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def _macd_signal(series: pd.Series,
                 fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD minus Signal line."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig  = macd.ewm(span=signal, adjust=False).mean()
    return macd - sig


def _consecutive_red_days(returns: pd.Series) -> pd.Series:
    """Count of consecutive trading days with negative return, reset on green day."""
    neg = (returns < 0).astype(int)
    groups = neg.ne(neg.shift()).cumsum()
    counts = neg.groupby(groups).cumsum()
    return counts


def _bollinger_position(prices: pd.Series, window: int = 20,
                        num_std: float = 2.0) -> pd.Series:
    """Position of current price within Bollinger Bands, scaled 0-1."""
    sma   = prices.rolling(window).mean()
    std   = prices.rolling(window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    width = (upper - lower).replace(0, np.nan)
    return ((prices - lower) / width).clip(0, 1).fillna(0.5)


def engineer_features(nifty_df: pd.DataFrame,
                      aux_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full 15-feature matrix from Nifty OHLCV + auxiliary data.

    Parameters
    ----------
    nifty_df : DataFrame with columns [Open, High, Low, Close, Volume]
    aux_df   : DataFrame with columns [vix_level, crude_close] (may be partial)

    Returns
    -------
    DataFrame with columns listed in ``TRAINING_FEATURES``, indexed by date.
    """
    logger.info("=" * 60)
    logger.info("STEP 2 / 7 — Engineering 15 features")
    logger.info("=" * 60)

    close: pd.Series = nifty_df["Close"].squeeze()
    volume: pd.Series = nifty_df["Volume"].squeeze()

    feat = pd.DataFrame(index=nifty_df.index)

    # --- Returns -----------------------------------------------------------
    feat["daily_return"]  = safe_pct_change(close, 1)
    feat["weekly_return"] = safe_pct_change(close, 5)
    feat["monthly_return"] = safe_pct_change(close, 20)

    # --- Volatility --------------------------------------------------------
    feat["rolling_volatility_20d"] = compute_rolling_volatility(feat["daily_return"], 20)
    feat["rolling_volatility_50d"] = compute_rolling_volatility(feat["daily_return"], 50)

    # --- Drawdown from 52-week (252-day) peak ------------------------------
    rolling_max_252 = close.rolling(window=252, min_periods=1).max()
    feat["drawdown_from_peak"] = (close - rolling_max_252) / rolling_max_252

    # --- RSI 14 ------------------------------------------------------------
    feat["rsi_14"] = _rsi(close, 14)

    # --- MACD signal -------------------------------------------------------
    feat["macd_signal"] = _macd_signal(close)

    # --- VIX ---------------------------------------------------------------
    if "vix_level" in aux_df.columns and aux_df["vix_level"].notna().sum() > 100:
        feat["vix_level"] = aux_df["vix_level"].reindex(feat.index, method="ffill")
    else:
        # Synthetic VIX: scale 20d annualised vol to mimic VIX range (12-45)
        raw = feat["rolling_volatility_20d"].fillna(0.15)
        feat["vix_level"] = 12 + (raw - raw.min()) / (raw.max() - raw.min() + 1e-9) * 33
        logger.info("  → using synthetic VIX derived from Nifty 20d vol")

    # --- Crude oil daily % change ------------------------------------------
    if "crude_close" in aux_df.columns and aux_df["crude_close"].notna().sum() > 100:
        crude = aux_df["crude_close"].reindex(feat.index, method="ffill")
        feat["crude_oil_change"] = safe_pct_change(crude, 1)
    else:
        feat["crude_oil_change"] = 0.0

    # --- Consecutive red days ----------------------------------------------
    feat["consecutive_red_days"] = _consecutive_red_days(feat["daily_return"])

    # --- Distance from 200-DMA --------------------------------------------
    dma_200 = close.rolling(window=200, min_periods=1).mean()
    feat["distance_from_200dma"] = (close - dma_200) / dma_200

    # --- FII flow proxy (high volume + negative return ≈ FII selling) ------
    vol_z = (volume - volume.rolling(50).mean()) / volume.rolling(50).std().replace(0, 1)
    ret_sign = np.where(feat["daily_return"] < 0, -1, 1)
    feat["fii_flow_proxy"] = vol_z * ret_sign

    # --- Bollinger band position -------------------------------------------
    feat["bollinger_band_position"] = _bollinger_position(close, 20)

    # --- Crash severity score (composite) ----------------------------------
    dd_norm  = feat["drawdown_from_peak"].clip(-0.50, 0).abs() / 0.50       # 0-1
    vol_norm = (feat["rolling_volatility_20d"] /
                feat["rolling_volatility_20d"].rolling(252, min_periods=60).mean()).clip(0, 5) / 5
    red_norm = feat["consecutive_red_days"].clip(0, 10) / 10
    feat["crash_severity_score"] = (0.40 * dd_norm + 0.35 * vol_norm + 0.25 * red_norm)

    # --- Final cleanup -----------------------------------------------------
    feat = feat.ffill().bfill()

    # Validate all expected columns exist
    missing = [c for c in TRAINING_FEATURES if c not in feat.columns]
    if missing:
        raise RuntimeError(f"Feature engineering bug — missing columns: {missing}")

    feat = feat[TRAINING_FEATURES]   # enforce canonical order
    logger.info(f"  Feature matrix: {feat.shape[0]} rows × {feat.shape[1]} cols")
    logger.info(f"  Date range: {feat.index.min().date()} → {feat.index.max().date()}")
    for col in TRAINING_FEATURES:
        pct_nan = feat[col].isna().mean() * 100
        if pct_nan > 0:
            logger.warning(f"  ⚠ {col}: {pct_nan:.1f}% NaN")
    return feat


# ============================================================
# §3  TARGET VARIABLE
# ============================================================

def create_panic_labels(features: pd.DataFrame) -> pd.Series:
    """
    Label trading days as panic_likely=1 when ALL of:
      • drawdown from 52-wk high  >  10 %
      • 20d volatility  >  2× rolling mean volatility
      • consecutive red days  ≥  3

    These conditions capture the exact market state where retail SIP
    stoppage rates historically spike above 50 %.
    """
    logger.info("=" * 60)
    logger.info("STEP 3 / 7 — Creating panic labels")
    logger.info("=" * 60)

    dd   = features["drawdown_from_peak"]
    vol  = features["rolling_volatility_20d"]
    reds = features["consecutive_red_days"]

    vol_mean = vol.rolling(window=252, min_periods=60).mean()

    cond_drawdown = dd < DRAWDOWN_THRESHOLD             # e.g. < -0.10
    cond_vol      = vol > (VOL_MULTIPLE_THRESH * vol_mean)   # 2× average
    cond_streak   = reds >= RED_DAYS_THRESHOLD           # ≥ 3 days

    labels = (cond_drawdown & cond_vol & cond_streak).astype(int)
    labels.name = "panic_likely"

    n_panic = labels.sum()
    n_total = len(labels)
    logger.info(f"  Label distribution:  0 (normal) = {n_total - n_panic}  |  1 (panic) = {n_panic}")
    logger.info(f"  Panic ratio: {n_panic / n_total * 100:.2f}%")

    if n_panic == 0:
        logger.warning(
            "  ⚠ Zero panic labels found — loosening thresholds for demo viability…"
        )
        # Fallback: label top-5% worst crash_severity_score days as panic
        threshold = features["crash_severity_score"].quantile(0.95)
        labels = (features["crash_severity_score"] >= threshold).astype(int)
        labels.name = "panic_likely"
        logger.info(f"  Fallback labels: 1 (panic) = {labels.sum()}")

    return labels


# ============================================================
# §4  WALK-FORWARD CROSS-VALIDATION  (time-series safe)
# ============================================================

def walk_forward_cv(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
) -> List[dict]:
    """
    Walk-forward expanding-window cross-validation.

    Each fold:
      train = all data up to split point
      test  = next CV_TEST_MONTHS months

    SMOTE is applied *inside* each fold (no data leakage).

    Returns list of per-fold metrics dicts.
    """
    logger.info("=" * 60)
    logger.info("STEP 4 / 7 — Walk-forward cross-validation")
    logger.info("=" * 60)

    from imblearn.over_sampling import SMOTE

    min_train_date = X.index.min() + pd.DateOffset(years=CV_TRAIN_YEARS)
    max_date       = X.index.max()
    split_dates    = pd.date_range(
        start=min_train_date, end=max_date - pd.DateOffset(months=CV_TEST_MONTHS),
        freq=f"{CV_STEP_MONTHS}MS",
    )

    fold_results: list[dict] = []

    for i, split in enumerate(split_dates, 1):
        test_end = split + pd.DateOffset(months=CV_TEST_MONTHS)

        train_mask = X.index < split
        test_mask  = (X.index >= split) & (X.index < test_end)

        X_train, y_train = X.loc[train_mask], y.loc[train_mask]
        X_test,  y_test  = X.loc[test_mask],  y.loc[test_mask]

        if len(X_test) < 20 or y_train.sum() < 5:
            continue   # skip degenerate folds

        # Scale
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)

        # SMOTE on train only
        try:
            sm = SMOTE(random_state=42, k_neighbors=min(3, int(y_train.sum()) - 1))
            X_tr_s, y_tr_r = sm.fit_resample(X_tr_s, y_train)
        except ValueError:
            y_tr_r = y_train   # too few minority samples

        model = xgb.XGBClassifier(**params)
        model.fit(X_tr_s, y_tr_r, eval_set=[(X_te_s, y_test)], verbose=False)

        y_pred = model.predict(X_te_s)
        y_prob = model.predict_proba(X_te_s)[:, 1]

        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = float("nan")

        f1 = f1_score(y_test, y_pred, zero_division=0)

        fold_results.append({
            "fold":           i,
            "train_end":      str(split.date()),
            "test_start":     str(split.date()),
            "test_end":       str(min(test_end, max_date).date()),
            "train_samples":  int(train_mask.sum()),
            "test_samples":   int(test_mask.sum()),
            "test_panic_pct": float(y_test.mean() * 100),
            "auroc":          round(auc, 4),
            "f1_panic":       round(f1, 4),
        })

        logger.info(
            f"  Fold {i:>2d}  |  train→{split.date()}  test→{min(test_end, max_date).date()}"
            f"  |  AUC={auc:.3f}  F1={f1:.3f}  (panic={y_test.mean()*100:.1f}%)"
        )

    if fold_results:
        avg_auc = np.nanmean([f["auroc"] for f in fold_results])
        avg_f1  = np.nanmean([f["f1_panic"] for f in fold_results])
        logger.info(f"\n  Mean AUROC = {avg_auc:.4f}   |   Mean F1 (panic) = {avg_f1:.4f}")
    else:
        logger.warning("  No valid CV folds produced — data may be too short.")

    return fold_results


# ============================================================
# §5  FINAL MODEL TRAINING
# ============================================================

def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    params: Optional[dict] = None,
) -> Tuple[xgb.XGBClassifier, StandardScaler]:
    """
    Train the production model on *all* available data with SMOTE.

    Returns (model, scaler).
    """
    from imblearn.over_sampling import SMOTE

    logger.info("=" * 60)
    logger.info("STEP 5 / 7 — Training final production model")
    logger.info("=" * 60)

    if params is None:
        params = dict(XGB_PARAMS)

    # Auto scale_pos_weight if not explicitly set
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    if n_pos > 0:
        auto_spw = round(n_neg / n_pos, 2)
        params.setdefault("scale_pos_weight", auto_spw)
        logger.info(f"  scale_pos_weight (auto) = {auto_spw}")
    params["max_depth"] = 6          # user spec override

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    try:
        k = min(5, n_pos - 1) if n_pos > 1 else 1
        sm = SMOTE(random_state=42, k_neighbors=k)
        X_resampled, y_resampled = sm.fit_resample(X_scaled, y)
        logger.info(
            f"  SMOTE resampling: {len(X)} → {len(X_resampled)} samples  "
            f"(0={int((y_resampled == 0).sum())}, 1={int((y_resampled == 1).sum())})"
        )
    except ValueError as e:
        logger.warning(f"  SMOTE failed ({e}), training on raw imbalanced data")
        X_resampled, y_resampled = X_scaled, y

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_resampled, y_resampled,
        eval_set=[(X_scaled, y)],
        verbose=False,
    )
    logger.info("  ✅ Final model training complete")

    # --- Evaluate on full data (train set — for reporting, CV gives true OOS) --
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    print("\n" + "=" * 64)
    print("   PanicGuard AI — Final Model Evaluation (full dataset)")
    print("=" * 64)
    print(classification_report(y, y_pred, target_names=["Normal", "Panic"], zero_division=0))
    try:
        auc = roc_auc_score(y, y_prob)
        print(f"   ROC-AUC: {auc:.4f}")
    except ValueError:
        auc = float("nan")
        print("   ROC-AUC: N/A (single class)")
    print("\n   Confusion Matrix:")
    cm = confusion_matrix(y, y_pred)
    print(f"   {cm}")
    print("=" * 64)

    # Feature importance
    imp = pd.Series(model.feature_importances_, index=TRAINING_FEATURES).sort_values(ascending=False)
    print("\n   Top Feature Importances:")
    for feat_name, importance in imp.head(10).items():
        bar = "█" * int(importance * 50)
        print(f"     {feat_name:<28s}  {importance:.4f}  {bar}")
    print()

    return model, scaler


# ============================================================
# §6  SHAP EXPLANATIONS
# ============================================================

def compute_shap_explanations(
    model: xgb.XGBClassifier,
    scaler: StandardScaler,
    X: pd.DataFrame,
) -> shap.TreeExplainer:
    """
    Build a SHAP TreeExplainer and generate a global summary bar plot.

    Returns the explainer for downstream per-sample explanations.
    """
    logger.info("=" * 60)
    logger.info("STEP 6 / 7 — SHAP explainability")
    logger.info("=" * 60)

    explainer = shap.TreeExplainer(model)
    X_scaled  = scaler.transform(X)

    # Compute SHAP values (for panic class)
    shap_values = explainer.shap_values(X_scaled)
    if isinstance(shap_values, list):
        sv = shap_values[1]     # class-1 (panic)
    else:
        sv = shap_values

    # --- Global summary bar plot -------------------------------------------
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig_summary, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(sv, X_scaled, feature_names=TRAINING_FEATURES,
                      plot_type="bar", show=False, max_display=15)
    plt.title("PanicGuard AI — SHAP Feature Importance (Global)", fontsize=13, pad=12)
    plt.tight_layout()
    summary_path = PLOTS_DIR / "shap_global_importance.png"
    fig_summary.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close(fig_summary)
    logger.info(f"  Saved global SHAP plot → {summary_path}")

    # --- Dot (beeswarm) plot -----------------------------------------------
    fig_dot, ax2 = plt.subplots(figsize=(10, 7))
    shap.summary_plot(sv, X_scaled, feature_names=TRAINING_FEATURES,
                      show=False, max_display=15)
    plt.title("PanicGuard AI — SHAP Feature Impact (Beeswarm)", fontsize=13, pad=12)
    plt.tight_layout()
    beeswarm_path = PLOTS_DIR / "shap_beeswarm.png"
    fig_dot.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
    plt.close(fig_dot)
    logger.info(f"  Saved beeswarm SHAP plot → {beeswarm_path}")

    return explainer


def generate_shap_waterfall(
    explainer: shap.TreeExplainer,
    scaler: StandardScaler,
    feature_row: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> Tuple[plt.Figure, list[dict]]:
    """
    Create a SHAP waterfall plot for a single observation.

    Returns (matplotlib Figure, top-K attributions as list of dicts).
    """
    x_scaled = scaler.transform(feature_row[TRAINING_FEATURES])
    sv = explainer.shap_values(x_scaled)
    if isinstance(sv, list):
        sv = sv[1]

    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = base_val[1]

    explanation = shap.Explanation(
        values=sv[0],
        base_values=float(base_val),
        data=x_scaled[0],
        feature_names=TRAINING_FEATURES,
    )

    fig = plt.figure(figsize=(10, 7))
    shap.plots.waterfall(explanation, max_display=SHAP_TOP_K_FEATURES, show=False)
    plt.title("PanicGuard AI — What's driving the panic score?", fontsize=12, pad=10)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Waterfall plot saved → {save_path}")

    # Top-K attributions
    pairs = sorted(zip(TRAINING_FEATURES, sv[0]), key=lambda t: abs(t[1]), reverse=True)
    top_k = [
        {"feature": f, "shap_value": round(float(v), 4), "abs_impact": round(abs(float(v)), 4)}
        for f, v in pairs[:SHAP_TOP_K_FEATURES]
    ]
    return fig, top_k


# ============================================================
# §7  PREDICTION FUNCTION
# ============================================================

_HUMAN_NAMES: dict[str, str] = {
    "daily_return":            "Daily return",
    "rolling_volatility_20d":  "20-day volatility",
    "rolling_volatility_50d":  "50-day volatility",
    "drawdown_from_peak":      "Drawdown from peak",
    "rsi_14":                  "RSI (14-day)",
    "macd_signal":             "MACD signal",
    "vix_level":               "India VIX level",
    "crude_oil_change":        "Crude oil change",
    "consecutive_red_days":    "Consecutive red days",
    "distance_from_200dma":    "Distance from 200-DMA",
    "weekly_return":           "Weekly return",
    "monthly_return":          "Monthly return",
    "fii_flow_proxy":          "FII flow proxy",
    "bollinger_band_position": "Bollinger band position",
    "crash_severity_score":    "Crash severity score",
}

_FACTOR_FORMATTERS: dict[str, Any] = {
    "drawdown_from_peak":      lambda v: f"{v*100:.1f}%",
    "rolling_volatility_20d":  lambda v: f"{v*100:.1f}% annualised",
    "rolling_volatility_50d":  lambda v: f"{v*100:.1f}% annualised",
    "daily_return":            lambda v: f"{v*100:.2f}%",
    "weekly_return":           lambda v: f"{v*100:.2f}%",
    "monthly_return":          lambda v: f"{v*100:.2f}%",
    "rsi_14":                  lambda v: f"{v:.1f}",
    "vix_level":               lambda v: f"{v:.1f}",
    "crude_oil_change":        lambda v: f"{v*100:.2f}%",
    "consecutive_red_days":    lambda v: f"{int(v)} days",
    "distance_from_200dma":    lambda v: f"{v*100:.1f}%",
    "fii_flow_proxy":          lambda v: f"{v:.2f} (z-score)",
    "bollinger_band_position": lambda v: f"{v:.2f} (0=lower, 1=upper)",
    "crash_severity_score":    lambda v: f"{v:.2f} / 1.00",
    "macd_signal":             lambda v: f"{v:.2f}",
}


def predict_panic_score(
    current_market_data: dict[str, float],
    model: Optional[xgb.XGBClassifier] = None,
    scaler: Optional[StandardScaler] = None,
    explainer: Optional[shap.TreeExplainer] = None,
) -> dict:
    """
    Run inference on a single observation and return a rich result dict.

    Parameters
    ----------
    current_market_data : dict
        Keys must match ``TRAINING_FEATURES`` values.
    model, scaler, explainer : optional
        Pre-loaded artefacts.  If None, loads from disk.

    Returns
    -------
    dict with keys:
        panic_score, risk_level, top_factors, explanation,
        recommendation, shap_plot_path
    """
    # --- Load model if needed ------------------------------------------------
    if model is None or scaler is None:
        model, scaler = _load_model_and_scaler()
    if explainer is None:
        explainer = shap.TreeExplainer(model)

    # --- Build feature row ---------------------------------------------------
    row_df = pd.DataFrame([current_market_data])[TRAINING_FEATURES]
    x_scaled = scaler.transform(row_df)

    # --- Predict -------------------------------------------------------------
    prob = float(model.predict_proba(x_scaled)[0][1])
    score = int(round(prob * 100))

    # --- Risk level ----------------------------------------------------------
    if score < 30:
        risk_level = "LOW"
        recommendation = "STAY_CALM"
    elif score < 60:
        risk_level = "MEDIUM"
        recommendation = "HOLD"
    elif score < 80:
        risk_level = "HIGH"
        recommendation = "HOLD"
    else:
        risk_level = "CRITICAL"
        recommendation = "INCREASE_SIP"

    # Additional logic: if VIX elevated but drawdown small → REBALANCE
    if (current_market_data.get("vix_level", 0) > 25 and
        abs(current_market_data.get("drawdown_from_peak", 0)) < 0.05):
        recommendation = "REBALANCE"

    # --- SHAP attributions ---------------------------------------------------
    sv = explainer.shap_values(x_scaled)
    if isinstance(sv, list):
        sv = sv[1]

    total_abs = np.sum(np.abs(sv[0])) or 1.0

    pairs = sorted(zip(TRAINING_FEATURES, sv[0]), key=lambda t: abs(t[1]), reverse=True)
    top_factors = []
    for feat, shap_val in pairs[:3]:
        raw_value = current_market_data.get(feat, 0.0)
        fmt = _FACTOR_FORMATTERS.get(feat, lambda v: f"{v:.4f}")
        top_factors.append({
            "factor":  _HUMAN_NAMES.get(feat, feat),
            "value":   fmt(raw_value),
            "impact":  f"{abs(shap_val) / total_abs * 100:.0f}%",
            "raw_shap": round(float(shap_val), 4),
        })

    # --- Human explanation ---------------------------------------------------
    parts = []
    for tf in top_factors:
        parts.append(f"{tf['factor']} is {tf['value']} (contributing {tf['impact']} to panic score)")
    explanation = "Market analysis: " + ", ".join(parts) + "."

    # --- Waterfall plot ------------------------------------------------------
    plot_path = PLOTS_DIR / "shap_waterfall_latest.png"
    fig, _ = generate_shap_waterfall(explainer, scaler, row_df, save_path=plot_path)
    plt.close(fig)

    return {
        "panic_score":    score,
        "risk_level":     risk_level,
        "top_factors":    top_factors,
        "explanation":    explanation,
        "recommendation": recommendation,
        "shap_plot_path": str(plot_path),
    }


# ============================================================
# §8  BACKTESTING
# ============================================================

def run_backtest(
    features: pd.DataFrame,
    model: xgb.XGBClassifier,
    scaler: StandardScaler,
) -> list[dict]:
    """
    Evaluate the model against known crash events.

    For each event, reports:
      • first detection date (score ≥ 60) relative to peak panic
      • early-warning lead time
      • hold vs. panic-sell outcome
    """
    logger.info("=" * 60)
    logger.info("STEP 7 / 7 — Backtesting against historical crashes")
    logger.info("=" * 60)

    results: list[dict] = []
    X_full_scaled = scaler.transform(features[TRAINING_FEATURES])
    probs = model.predict_proba(X_full_scaled)[:, 1]
    score_series = pd.Series(probs * 100, index=features.index, name="panic_score")

    for event in BACKTEST_EVENTS:
        name       = event["name"]
        start      = pd.Timestamp(event["start"])
        peak_panic = pd.Timestamp(event["peak_panic"])
        end        = pd.Timestamp(event["end"])

        # Clip to available data
        mask = (features.index >= start) & (features.index <= end)
        if mask.sum() == 0:
            logger.warning(f"  {name}: no data in range, skipping")
            results.append({"event": name, "status": "NO_DATA"})
            continue

        window_scores = score_series.loc[mask]
        max_score     = window_scores.max()
        max_score_date = window_scores.idxmax()

        # First detection: first day score crosses 60
        alerts = window_scores[window_scores >= 60]
        if len(alerts) > 0:
            first_alert = alerts.index[0]
            lead_days   = (peak_panic - first_alert).days
            detected    = True
        else:
            first_alert = None
            lead_days   = 0
            detected    = False

        result = {
            "event":            name,
            "detected":         detected,
            "peak_panic_score": round(float(max_score), 1),
            "peak_score_date":  str(max_score_date.date()) if pd.notna(max_score_date) else "N/A",
            "first_alert_date": str(first_alert.date()) if first_alert else "N/A",
            "lead_time_days":   lead_days,
            "hold_gain_pct":    event["hold_gain"],
            "sell_loss_pct":    event["sell_loss"],
        }
        results.append(result)

        # Pretty-print
        status = "✅ DETECTED" if detected else "❌ MISSED"
        print(f"\n  {'─'*56}")
        print(f"  {status}  {name}")
        print(f"  {'─'*56}")
        print(f"    Peak panic score : {max_score:.1f}/100  on {result['peak_score_date']}")
        if detected:
            print(f"    First alert      : {result['first_alert_date']}  ({lead_days} days before peak panic)")
            print(f"    → PanicGuard would have alerted investors {lead_days} days early")
        if event["hold_gain"] is not None:
            print(f"    HOLD outcome     : +{event['hold_gain']:.0f}% gain")
            print(f"    PANIC-SELL loss   : {event['sell_loss']:.0f}% realised loss")
            spread = event["hold_gain"] - event["sell_loss"]
            print(f"    ▸ PanicGuard advantage: {spread:.0f} pp difference")
        else:
            print(f"    (Event ongoing — outcome TBD)")

    return results


# ============================================================
# §9  SAVE ARTIFACTS
# ============================================================

def save_pipeline_artifacts(
    model: xgb.XGBClassifier,
    scaler: StandardScaler,
    cv_results: list[dict],
    backtest_results: list[dict],
) -> None:
    """Persist model, scaler, feature list, and metrics to models/ directory."""
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # Model & scaler
    model_path  = Path(MODEL_SAVE_PATH)
    scaler_path = Path(SCALER_SAVE_PATH)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    logger.info(f"  Model  → {model_path}")
    logger.info(f"  Scaler → {scaler_path}")

    # Also save alongside as panic_model.joblib (user-specified name)
    alt_path = MODELS_DIR / "panic_model.joblib"
    joblib.dump(model, alt_path)
    logger.info(f"  Model  → {alt_path}  (alias)")

    # Feature column list (for downstream consumers)
    feat_path = SAVED_MODELS_DIR / "feature_columns.json"
    with open(feat_path, "w") as f:
        json.dump(TRAINING_FEATURES, f, indent=2)
    logger.info(f"  Features → {feat_path}")

    # CV metrics
    cv_path = METRICS_DIR / "cv_results.json"
    with open(cv_path, "w") as f:
        json.dump(cv_results, f, indent=2, default=str)
    logger.info(f"  CV metrics → {cv_path}")

    # Backtest metrics
    bt_path = METRICS_DIR / "backtest_results.json"
    with open(bt_path, "w") as f:
        json.dump(backtest_results, f, indent=2, default=str)
    logger.info(f"  Backtest results → {bt_path}")


# ============================================================
# §10  INTERNAL HELPERS
# ============================================================

def _load_model_and_scaler() -> Tuple[xgb.XGBClassifier, StandardScaler]:
    """Load persisted model and scaler, or raise FileNotFoundError."""
    m_path = Path(MODEL_SAVE_PATH)
    s_path = Path(SCALER_SAVE_PATH)
    if not m_path.exists():
        raise FileNotFoundError(f"Model not found at {m_path}. Run the training pipeline first.")
    if not s_path.exists():
        raise FileNotFoundError(f"Scaler not found at {s_path}.")
    return joblib.load(m_path), joblib.load(s_path)


# ============================================================
# §11  MAIN PIPELINE
# ============================================================

def main() -> None:
    """
    Full end-to-end pipeline:
        fetch → engineer → label → CV → train → SHAP → backtest → save
    """
    pipeline_start = datetime.now()
    print()
    print("╔" + "═" * 62 + "╗")
    print("║   🛡️  PanicGuard AI — ML Training Pipeline                   ║")
    print("║   Timestamp: " + pipeline_start.strftime("%Y-%m-%d %H:%M:%S") + "                            ║")
    print("╚" + "═" * 62 + "╝")
    print()

    # ------------------------------------------------------------------
    # 1. Fetch data
    # ------------------------------------------------------------------
    try:
        nifty_df, aux_df = fetch_training_data()
    except Exception as e:
        logger.error(f"Live data fetch failed: {e}")
        logger.info("Falling back to demo (synthetic) data…")
        nifty_df, aux_df = _generate_demo_data()

    # ------------------------------------------------------------------
    # 2. Feature engineering
    # ------------------------------------------------------------------
    features = engineer_features(nifty_df, aux_df)

    # ------------------------------------------------------------------
    # 3. Create labels
    # ------------------------------------------------------------------
    labels = create_panic_labels(features)

    # Drop NaN rows  (leading window warmup)
    valid_mask = features.notna().all(axis=1) & labels.notna()
    X = features.loc[valid_mask]
    y = labels.loc[valid_mask]
    logger.info(f"  Clean dataset: {len(X)} rows  (dropped {len(features) - len(X)} warmup rows)")

    # ------------------------------------------------------------------
    # 4. Walk-forward CV
    # ------------------------------------------------------------------
    cv_results = walk_forward_cv(X, y, params=dict(XGB_PARAMS, max_depth=6))

    # ------------------------------------------------------------------
    # 5. Train final model on all data
    # ------------------------------------------------------------------
    model, scaler = train_final_model(X, y, params=dict(XGB_PARAMS))

    # ------------------------------------------------------------------
    # 6. SHAP
    # ------------------------------------------------------------------
    explainer = compute_shap_explanations(model, scaler, X)

    # Generate waterfall for the most recent observation
    latest_row = X.tail(1)
    waterfall_path = PLOTS_DIR / "shap_waterfall_latest.png"
    fig, top_attrs = generate_shap_waterfall(explainer, scaler, latest_row, save_path=waterfall_path)
    plt.close(fig)
    logger.info(f"  Latest top-3 drivers: {top_attrs[:3]}")

    # Quick demo of the predict function
    print("\n" + "─" * 64)
    print("  Demo prediction (latest market data):")
    print("─" * 64)
    pred = predict_panic_score(
        current_market_data=latest_row.iloc[0].to_dict(),
        model=model,
        scaler=scaler,
        explainer=explainer,
    )
    print(f"  Panic Score : {pred['panic_score']}/100  ({pred['risk_level']})")
    print(f"  Recommendation : {pred['recommendation']}")
    for tf in pred["top_factors"]:
        print(f"    • {tf['factor']}: {tf['value']}  ({tf['impact']} impact)")
    print(f"  Explanation : {pred['explanation']}")
    print()

    # ------------------------------------------------------------------
    # 7. Backtest
    # ------------------------------------------------------------------
    bt_results = run_backtest(features, model, scaler)

    # ------------------------------------------------------------------
    # 8. Save everything
    # ------------------------------------------------------------------
    save_pipeline_artifacts(model, scaler, cv_results, bt_results)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = (datetime.now() - pipeline_start).total_seconds()
    print()
    print("╔" + "═" * 62 + "╗")
    print("║   ✅  Pipeline complete                                       ║")
    print(f"║   Runtime: {elapsed:.1f}s" + " " * (51 - len(f"{elapsed:.1f}s")) + "║")
    print("║                                                              ║")
    print("║   Artifacts saved:                                           ║")
    print(f"║     • {MODEL_SAVE_PATH:<54s} ║")
    print(f"║     • {SCALER_SAVE_PATH:<54s} ║")
    print(f"║     • models/panic_model.joblib                              ║")
    print(f"║     • models/plots/shap_*.png                                ║")
    print(f"║     • models/metrics/cv_results.json                         ║")
    print(f"║     • models/metrics/backtest_results.json                   ║")
    print("╚" + "═" * 62 + "╝")
    print()


if __name__ == "__main__":
    main()
