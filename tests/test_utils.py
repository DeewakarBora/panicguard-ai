"""
Tests: Crisis Detector
"""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from utils.helpers import classify_panic_score, format_inr, compute_rolling_volatility, safe_pct_change


class TestClassifyPanicScore:
    def test_normal_regime(self):
        result = classify_panic_score(0.10)
        assert result["label"] == "Normal Market"
        assert result["emoji"] == "✅"

    def test_elevated_regime(self):
        result = classify_panic_score(0.45)
        assert result["label"] == "Elevated Stress"
        assert result["emoji"] == "⚠️"

    def test_crisis_regime(self):
        result = classify_panic_score(0.70)
        assert result["label"] == "Crisis Mode"
        assert result["emoji"] == "🔴"

    def test_extreme_regime(self):
        result = classify_panic_score(0.90)
        assert result["label"] == "Extreme Panic"
        assert result["emoji"] == "🚨"

    def test_boundary_low(self):
        result = classify_panic_score(0.30)
        assert result["label"] == "Elevated Stress"

    def test_boundary_zero(self):
        result = classify_panic_score(0.0)
        assert result["label"] == "Normal Market"

    def test_boundary_one(self):
        result = classify_panic_score(1.0)
        assert result["label"] == "Extreme Panic"


class TestFormatInr:
    def test_crore(self):
        assert "Cr" in format_inr(1_50_00_000)

    def test_lakh(self):
        assert "L" in format_inr(5_00_000)

    def test_small(self):
        assert "₹" in format_inr(5000)


class TestFinancialHelpers:
    def test_safe_pct_change_no_div_zero(self):
        s = pd.Series([0.0, 0.0, 1.0, 2.0])
        result = safe_pct_change(s, 1)
        assert not result.isnull().any()

    def test_rolling_volatility_shape(self):
        returns = pd.Series(np.random.randn(100) * 0.01)
        vol = compute_rolling_volatility(returns, window=20)
        assert len(vol) == len(returns)
