"""
Agent 1: Crisis Detector  —  Market Watcher
============================================
Loads the trained XGBoost panic-detection model, fetches live market data,
engineers the exact 15 features the model was trained on, and produces a
rich panic assessment with SHAP explanations.

Integrates with:
    - models/train_panic_model.py   (TRAINING_FEATURES, engineer_features,
                                     predict_panic_score, generate_shap_waterfall)
    - data/fetch_market_data.py     (fetch_ticker_history)
    - data/historical_crashes.py    (HISTORICAL_CRASHES, CrashEvent)
    - utils/config.py               (thresholds, tickers)
    - utils/helpers.py              (logging, retry, formatting)

Usage:
    detector = CrisisDetector()
    result   = detector.scan_market()
    if detector.should_alert():
        print("ALERT — market panic detected")
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.config import (
    CRUDE_TICKER,
    INDIA_VIX_TICKER,
    MODEL_SAVE_PATH,
    NIFTY50_TICKER,
    PANIC_THRESHOLD_HIGH,
    PANIC_THRESHOLD_LOW,
    PANIC_THRESHOLD_MEDIUM,
    SCALER_SAVE_PATH,
    SHAP_TOP_K_FEATURES,
)
from utils.helpers import (
    compute_rolling_volatility,
    format_inr,
    get_logger,
    retry,
    safe_pct_change,
)
from data.historical_crashes import HISTORICAL_CRASHES, CrashEvent, get_all_crashes

# Import the canonical feature list + engineering function from the training module
from models.train_panic_model import (
    TRAINING_FEATURES,
    engineer_features,
    predict_panic_score as _model_predict,
)

logger = get_logger("agents.crisis_detector")


# ============================================================
# Demo-mode fallback data  (April 2026 tariff crash scenario)
# ============================================================

_DEMO_FEATURES: dict[str, float] = {
    "daily_return":            -0.0285,
    "rolling_volatility_20d":   0.312,
    "rolling_volatility_50d":   0.245,
    "drawdown_from_peak":      -0.122,
    "rsi_14":                   28.4,
    "macd_signal":             -185.0,
    "vix_level":                23.6,
    "crude_oil_change":        -0.032,
    "consecutive_red_days":     5.0,
    "distance_from_200dma":    -0.074,
    "weekly_return":           -0.068,
    "monthly_return":          -0.115,
    "fii_flow_proxy":          -2.35,
    "bollinger_band_position":  0.08,
    "crash_severity_score":     0.72,
}

_DEMO_MARKET_SUMMARY = {
    "nifty":        21_843.50,
    "nifty_change": -2.85,
    "vix":          23.6,
    "crude":        61.20,
}


class CrisisDetector:
    """
    Real-time market panic regime classifier.

    Loads the trained XGBoost model, fetches live market data, engineers
    features, and returns a structured panic assessment.
    """

    def __init__(self) -> None:
        self.model = None
        self.scaler = None
        self._demo_mode = False
        self._last_result: Optional[dict] = None
        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the trained XGBoost model and scaler from disk."""
        model_path  = Path(MODEL_SAVE_PATH)
        scaler_path = Path(SCALER_SAVE_PATH)
        try:
            if model_path.exists() and scaler_path.exists():
                self.model  = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                logger.info(f"Model loaded from {model_path}")
            else:
                logger.warning("Trained model not found — using demo mode.")
                self._demo_mode = True
        except Exception as e:
            logger.error(f"Model load error: {e} — falling back to demo mode.")
            self._demo_mode = True

    # ------------------------------------------------------------------
    # Live data fetching + feature engineering
    # ------------------------------------------------------------------

    def _fetch_live_features(self) -> Tuple[pd.DataFrame, dict]:
        """
        Fetch recent market data and engineer the exact 15 features
        the model was trained on.

        Returns
        -------
        features : single-row DataFrame with TRAINING_FEATURES columns
        market_summary : dict with nifty, nifty_change, vix, crude
        """
        import yfinance as yf

        logger.info("Fetching live market data…")

        # --- Nifty 50 OHLCV ------------------------------------------------
        nifty_df = yf.download(NIFTY50_TICKER, period="1y",
                               auto_adjust=True, progress=False)
        if isinstance(nifty_df.columns, pd.MultiIndex):
            nifty_df.columns = nifty_df.columns.get_level_values(0)
        if nifty_df.empty:
            raise ValueError("Nifty data fetch returned empty DataFrame")

        latest_close  = float(nifty_df["Close"].iloc[-1])
        prev_close    = float(nifty_df["Close"].iloc[-2]) if len(nifty_df) > 1 else latest_close
        nifty_change  = (latest_close - prev_close) / prev_close * 100

        # --- India VIX (optional) -------------------------------------------
        try:
            vix_df = yf.download(INDIA_VIX_TICKER, period="1y",
                                 auto_adjust=True, progress=False)
            if isinstance(vix_df.columns, pd.MultiIndex):
                vix_df.columns = vix_df.columns.get_level_values(0)
            vix_series = vix_df["Close"].rename("vix_level")
            vix_current = float(vix_series.iloc[-1])
        except Exception:
            vix_series = None
            vix_current = None

        # --- Crude Oil (optional) -------------------------------------------
        try:
            crude_df = yf.download(CRUDE_TICKER, period="1y",
                                   auto_adjust=True, progress=False)
            if isinstance(crude_df.columns, pd.MultiIndex):
                crude_df.columns = crude_df.columns.get_level_values(0)
            crude_series = crude_df["Close"].rename("crude_close")
            crude_current = float(crude_series.iloc[-1])
        except Exception:
            crude_series = pd.Series(dtype=float, name="crude_close")
            crude_current = None

        # --- Build aux DataFrame for engineer_features() --------------------
        aux = pd.DataFrame(index=nifty_df.index)
        if vix_series is not None:
            aux["vix_level"] = vix_series.reindex(aux.index, method="ffill")
        if not crude_series.empty:
            aux["crude_close"] = crude_series.reindex(aux.index, method="ffill")
        aux = aux.ffill().bfill()

        # --- Engineer features (uses the *same* function as training) -------
        full_features = engineer_features(nifty_df, aux)
        latest_row    = full_features.tail(1)

        market_summary = {
            "nifty":        round(latest_close, 2),
            "nifty_change": round(nifty_change, 2),
            "vix":          round(vix_current, 1) if vix_current else None,
            "crude":        round(crude_current, 2) if crude_current else None,
        }

        return latest_row, market_summary

    # ------------------------------------------------------------------
    # Scan market (main entry point)
    # ------------------------------------------------------------------

    def scan_market(self) -> dict:
        """
        Run a full crisis scan: fetch data → predict → explain.

        Returns
        -------
        dict with keys:
            panic_score, risk_level, top_factors, shap_explanation,
            market_summary, timestamp, is_demo
        """
        logger.info("="*50)
        logger.info("CrisisDetector.scan_market()")
        logger.info("="*50)
        timestamp = datetime.now()

        # --- Attempt live data, fall back to demo --------------------------
        try:
            if self._demo_mode:
                raise RuntimeError("Demo-mode forced")

            latest_row, market_summary = self._fetch_live_features()
            feature_dict = latest_row.iloc[0].to_dict()
            is_demo = False

        except Exception as e:
            logger.warning(f"Live data failed ({e}) — using demo fallback.")
            feature_dict   = dict(_DEMO_FEATURES)
            market_summary = dict(_DEMO_MARKET_SUMMARY)
            is_demo = True

        # --- Run prediction --------------------------------------------------
        if self.model is not None and self.scaler is not None:
            pred = _model_predict(
                current_market_data=feature_dict,
                model=self.model,
                scaler=self.scaler,
            )
        else:
            # Rule-based fallback when no model is available
            pred = self._rule_based_prediction(feature_dict)

        # --- Build SHAP explanation string ---------------------------------
        shap_parts = []
        for tf in pred.get("top_factors", []):
            shap_parts.append(
                f"{tf['factor']} is {tf['value']} (contributing {tf['impact']} to panic score)"
            )
        shap_explanation = ". ".join(shap_parts) + "." if shap_parts else pred.get("explanation", "")

        result = {
            "panic_score":      pred["panic_score"],
            "risk_level":       pred["risk_level"],
            "top_factors":      pred.get("top_factors", []),
            "shap_explanation": shap_explanation,
            "recommendation":   pred.get("recommendation", "HOLD"),
            "market_summary":   market_summary,
            "raw_features":     feature_dict,
            "shap_plot_path":   pred.get("shap_plot_path"),
            "timestamp":        timestamp.isoformat(),
            "is_demo":          is_demo,
        }

        self._last_result = result
        logger.info(
            f"Panic Score: {result['panic_score']}/100  "
            f"Risk: {result['risk_level']}  Demo: {is_demo}"
        )
        return result

    # ------------------------------------------------------------------
    # Historical crash comparison
    # ------------------------------------------------------------------

    def get_crash_comparison(self) -> dict:
        """
        Compare current market conditions against all historical crashes
        using Euclidean distance on normalised key features.

        Returns
        -------
        dict with keys:
            most_similar_crash, similarity_score, all_comparisons,
            recovery_timeline
        """
        logger.info("Running crash comparison…")

        # Get current features
        if self._last_result and "raw_features" in self._last_result:
            current = self._last_result["raw_features"]
        else:
            current = dict(_DEMO_FEATURES)

        # Key comparison dimensions + typical value ranges for normalisation
        compare_keys = {
            "drawdown_from_peak":      0.60,   # range: 0 to -60%
            "rolling_volatility_20d":  0.50,   # range: ~5% to 50%
            "consecutive_red_days":    10.0,    # range: 0 to 10
            "crash_severity_score":    1.0,     # range: 0 to 1
            "vix_level":               45.0,    # range: ~10 to 45
        }

        # Build a feature-signature for each historical crash
        crash_signatures: dict[str, dict[str, float]] = {
            "Global Financial Crisis (2008-09)": {
                "drawdown_from_peak": -0.609,
                "rolling_volatility_20d": 0.48,
                "consecutive_red_days": 7,
                "crash_severity_score": 0.92,
                "vix_level": 42.0,
            },
            "European Debt Crisis (2011)": {
                "drawdown_from_peak": -0.28,
                "rolling_volatility_20d": 0.28,
                "consecutive_red_days": 4,
                "crash_severity_score": 0.55,
                "vix_level": 30.0,
            },
            "China Slowdown (2015-16)": {
                "drawdown_from_peak": -0.22,
                "rolling_volatility_20d": 0.25,
                "consecutive_red_days": 3,
                "crash_severity_score": 0.45,
                "vix_level": 27.0,
            },
            "IL&FS Crisis (2018-19)": {
                "drawdown_from_peak": -0.16,
                "rolling_volatility_20d": 0.22,
                "consecutive_red_days": 3,
                "crash_severity_score": 0.38,
                "vix_level": 22.0,
            },
            "COVID-19 Crash (2020)": {
                "drawdown_from_peak": -0.384,
                "rolling_volatility_20d": 0.55,
                "consecutive_red_days": 8,
                "crash_severity_score": 0.88,
                "vix_level": 40.0,
            },
            "Rate Hike Selloff (2022)": {
                "drawdown_from_peak": -0.16,
                "rolling_volatility_20d": 0.20,
                "consecutive_red_days": 4,
                "crash_severity_score": 0.35,
                "vix_level": 24.0,
            },
        }

        recovery_data = {
            "Global Financial Crisis (2008-09)": {"months": 24, "post_gain_pct": 190},
            "European Debt Crisis (2011)":       {"months": 14, "post_gain_pct": 45},
            "China Slowdown (2015-16)":          {"months":  9, "post_gain_pct": 30},
            "IL&FS Crisis (2018-19)":            {"months":  8, "post_gain_pct": 25},
            "COVID-19 Crash (2020)":             {"months":  7, "post_gain_pct": 112},
            "Rate Hike Selloff (2022)":          {"months":  9, "post_gain_pct": 28},
        }

        # Compute normalised Euclidean distances
        comparisons = []
        for crash_name, sig in crash_signatures.items():
            dist_squared = 0.0
            for key, norm_range in compare_keys.items():
                cur_val   = float(current.get(key, 0))
                crash_val = float(sig.get(key, 0))
                dist_squared += ((cur_val - crash_val) / norm_range) ** 2
            distance = dist_squared ** 0.5
            similarity = max(0.0, 1.0 - distance / 3.0)  # scale to 0-1

            crash_event = None
            for c in HISTORICAL_CRASHES:
                if crash_name.split("(")[0].strip().lower() in c.name.lower():
                    crash_event = c
                    break

            comparisons.append({
                "crash_name":       crash_name,
                "distance":         round(distance, 4),
                "similarity_score": round(similarity, 3),
                "recovery_months":  recovery_data.get(crash_name, {}).get("months", 0),
                "post_gain_pct":    recovery_data.get(crash_name, {}).get("post_gain_pct", 0),
                "key_lesson":       crash_event.key_lesson if crash_event else "",
            })

        comparisons.sort(key=lambda x: x["distance"])
        best = comparisons[0]

        logger.info(f"Most similar crash: {best['crash_name']} (similarity={best['similarity_score']:.2f})")

        return {
            "most_similar_crash": best["crash_name"],
            "similarity_score":   best["similarity_score"],
            "recovery_months":    best["recovery_months"],
            "post_bottom_gain":   best["post_gain_pct"],
            "key_lesson":         best["key_lesson"],
            "all_comparisons":    comparisons,
            "recovery_timeline":  (
                f"In the {best['crash_name']}, the market fully recovered in "
                f"{best['recovery_months']} months and investors who held gained "
                f"{best['post_gain_pct']}% from the bottom."
            ),
        }

    # ------------------------------------------------------------------
    # Alert check
    # ------------------------------------------------------------------

    def should_alert(self) -> bool:
        """Return True if the latest panic score exceeds the medium threshold (60)."""
        if self._last_result is None:
            self.scan_market()
        score = self._last_result.get("panic_score", 0)
        threshold = int(PANIC_THRESHOLD_MEDIUM * 100)
        alert = score >= threshold
        if alert:
            logger.warning(f"ALERT triggered — panic_score={score} >= threshold={threshold}")
        return alert

    # ------------------------------------------------------------------
    # Rule-based fallback (no model available)
    # ------------------------------------------------------------------

    def _rule_based_prediction(self, features: dict) -> dict:
        """
        Heuristic panic score when the XGBoost model is unavailable.
        Weighted combination of drawdown, volatility, VIX, and red-day streak.
        """
        dd       = abs(features.get("drawdown_from_peak", 0))
        vol_20   = features.get("rolling_volatility_20d", 0.15)
        vix      = features.get("vix_level", 15)
        reds     = features.get("consecutive_red_days", 0)
        severity = features.get("crash_severity_score", 0)

        # Normalise each to 0-1
        dd_n    = min(dd / 0.40, 1.0)
        vol_n   = min(vol_20 / 0.50, 1.0)
        vix_n   = min(max(vix - 12, 0) / 33, 1.0)
        red_n   = min(reds / 8, 1.0)

        raw = 0.30 * dd_n + 0.25 * vol_n + 0.25 * vix_n + 0.20 * red_n
        score = int(round(raw * 100))

        if score < 30:
            risk = "LOW"
            rec  = "STAY_CALM"
        elif score < 60:
            risk = "MEDIUM"
            rec  = "HOLD"
        elif score < 80:
            risk = "HIGH"
            rec  = "HOLD"
        else:
            risk = "CRITICAL"
            rec  = "INCREASE_SIP"

        top_factors = sorted([
            {"factor": "Drawdown from peak",   "value": f"{dd*100:.1f}%",     "impact": f"{30}%"},
            {"factor": "Volatility (20d)",     "value": f"{vol_20*100:.1f}%", "impact": f"{25}%"},
            {"factor": "India VIX",            "value": f"{vix:.1f}",         "impact": f"{25}%"},
            {"factor": "Consecutive red days", "value": f"{int(reds)} days",  "impact": f"{20}%"},
        ], key=lambda x: int(x["impact"].replace("%", "")), reverse=True)

        explanation = (
            f"Market has dropped {dd*100:.1f}% from its 52-week peak. "
            f"Volatility is at {vol_20*100:.0f}% annualised. "
            f"VIX reads {vix:.1f}. {int(reds)} consecutive red days."
        )
        logger.info(f"Rule-based prediction: score={score}, risk={risk}")

        return {
            "panic_score":    score,
            "risk_level":     risk,
            "top_factors":    top_factors,
            "explanation":    explanation,
            "recommendation": rec,
            "shap_plot_path": None,
        }
