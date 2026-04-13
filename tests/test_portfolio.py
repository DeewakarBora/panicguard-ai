"""
Tests: Portfolio Analyzer
"""

import pytest
from agents.portfolio_analyzer import PortfolioAnalyzer

SAMPLE_PORTFOLIO = {
    "total_invested_inr": 200_000,
    "holdings": [
        {"name": "HDFC Mid Cap Fund", "type": "mutual_fund", "value_inr": 150_000},
        {"name": "RELIANCE.NS",       "type": "stock",       "value_inr": 70_000},
    ],
    "sips": [
        {"name": "Axis Bluechip Fund", "monthly_inr": 5_000},
    ],
}


class TestPortfolioAnalyzer:
    def setup_method(self):
        self.analyzer = PortfolioAnalyzer(SAMPLE_PORTFOLIO)

    def test_total_value(self):
        assert self.analyzer.total_value == 220_000

    def test_pnl_positive(self):
        pnl = self.analyzer.compute_unrealized_pnl()
        assert pnl["gain_loss_inr"] == 20_000
        assert pnl["gain_loss_pct"] == pytest.approx(10.0, 0.1)

    def test_sip_impact_structure(self):
        sip = self.analyzer.compute_sip_impact(panic_score=0.70)
        assert "total_monthly_sip_inr" in sip
        assert "cost_of_panic_pause_inr" in sip
        assert sip["total_monthly_sip_inr"] == 5_000
        assert sip["cost_of_panic_pause_inr"] > 0

    def test_sector_exposure_sums_to_100(self):
        exposure = self.analyzer.compute_sector_exposure()
        total_pct = sum(e["pct"] for e in exposure)
        assert abs(total_pct - 100.0) < 0.5

    def test_historical_analogue_extreme_panic(self):
        analogue = self.analyzer.get_historical_analogue(0.85)
        assert "COVID" in analogue["name"]

    def test_run_returns_expected_keys(self):
        result = self.analyzer.run(panic_score=0.72)
        for key in ["portfolio_value", "pnl", "sip_impact", "sector_exposure", "historical_analogue", "at_risk"]:
            assert key in result

    def test_at_risk_true_for_equity_in_crisis(self):
        result = self.analyzer.run(panic_score=0.75)
        assert result["at_risk"] is True

    def test_not_at_risk_in_normal_market(self):
        result = self.analyzer.run(panic_score=0.10)
        assert result["at_risk"] is False
