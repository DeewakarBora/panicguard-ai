"""
ML Module: Model Utilities
===========================
Shared helpers for model loading, feature validation,
SHAP visualization, and prediction wrappers.
"""

import os
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Streamlit compatibility

from utils.config import MODEL_SAVE_PATH, SCALER_SAVE_PATH, FEATURE_COLUMNS, SHAP_TOP_K_FEATURES
from utils.helpers import get_logger, classify_panic_score

logger = get_logger(__name__)


# ============================================================
# Model Loading
# ============================================================

def load_model_artifacts() -> tuple:
    """
    Load the trained model and scaler from disk.

    Returns:
        (model, scaler) or (None, None) if not found.
    """
    if not os.path.exists(MODEL_SAVE_PATH):
        logger.warning(f"Model not found at {MODEL_SAVE_PATH}")
        return None, None
    if not os.path.exists(SCALER_SAVE_PATH):
        logger.warning(f"Scaler not found at {SCALER_SAVE_PATH}")
        return None, None

    model = joblib.load(MODEL_SAVE_PATH)
    scaler = joblib.load(SCALER_SAVE_PATH)
    logger.info("Model artifacts loaded.")
    return model, scaler


# ============================================================
# Feature Validation
# ============================================================

def validate_features(df: pd.DataFrame) -> tuple[bool, list]:
    """
    Check if all required feature columns are present.

    Returns:
        (is_valid: bool, missing_features: list)
    """
    missing = [f for f in FEATURE_COLUMNS if f not in df.columns]
    return len(missing) == 0, missing


# ============================================================
# Prediction
# ============================================================

def predict_panic_score(model, scaler, feature_row: pd.DataFrame) -> float:
    """
    Run inference on a single feature row.

    Args:
        model: Trained XGBoost model.
        scaler: Fitted StandardScaler.
        feature_row: Single-row DataFrame with feature columns.

    Returns:
        Float panic score [0, 1].
    """
    x_scaled = scaler.transform(feature_row[FEATURE_COLUMNS])
    score = float(model.predict_proba(x_scaled)[0][1])
    return round(score, 4)


# ============================================================
# SHAP Visualization
# ============================================================

def generate_shap_waterfall_plot(model, scaler, feature_row: pd.DataFrame) -> plt.Figure:
    """
    Generate a SHAP waterfall plot for a single prediction.
    Returns a Matplotlib figure suitable for embedding in Streamlit.

    Args:
        model: Trained XGBoost model.
        scaler: Fitted StandardScaler.
        feature_row: Single-row DataFrame with feature columns.

    Returns:
        Matplotlib Figure object.
    """
    x = feature_row[FEATURE_COLUMNS]
    x_scaled = scaler.transform(x)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_scaled)

    # For binary XGB: shap_values is (n_samples, n_features)
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]

    # Build SHAP Explanation object
    explanation = shap.Explanation(
        values=sv,
        base_values=explainer.expected_value if not isinstance(explainer.expected_value, list)
                    else explainer.expected_value[1],
        data=x_scaled[0],
        feature_names=FEATURE_COLUMNS,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(explanation, max_display=SHAP_TOP_K_FEATURES, show=False)
    plt.tight_layout()
    return fig


def generate_shap_bar_dict(model, scaler, feature_row: pd.DataFrame) -> list[dict]:
    """
    Return top SHAP attributions as a list of dicts for Plotly rendering.

    Returns:
        [{"feature": str, "shap_value": float, "direction": "positive"|"negative"}, ...]
    """
    x = feature_row[FEATURE_COLUMNS]
    x_scaled = scaler.transform(x)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_scaled)

    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]

    pairs = sorted(zip(FEATURE_COLUMNS, sv), key=lambda t: abs(t[1]), reverse=True)
    return [
        {
            "feature": feat,
            "shap_value": round(float(val), 4),
            "direction": "positive" if val > 0 else "negative",
        }
        for feat, val in pairs[:SHAP_TOP_K_FEATURES]
    ]
