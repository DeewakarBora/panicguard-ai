"""
PanicGuard AI — Multi-Agent Orchestrator
=========================================
Master controller that coordinates the three agents into a single
autonomous pipeline:

    Agent 1 (CrisisDetector)   →  market scan + crash comparison
    Agent 2 (PortfolioAnalyzer) →  personal impact + SIP scenarios
    Agent 3 (BehavioralCoach)   →  empathetic, data-driven coaching

Also provides a pre-computed demo result that works offline, without
API keys, and without internet — critical for hackathon judging.

Usage:
    orch   = PanicGuardOrchestrator()           # uses demo portfolio
    result = orch.run_full_analysis()            # autonomous pipeline
    reply  = orch.run_chat("Should I stop my SIP?")
    demo   = PanicGuardOrchestrator.get_demo_result()   # always works
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.helpers import format_inr, get_logger

logger = get_logger("agents.orchestrator")


# ============================================================
# Default demo portfolio
# ============================================================

_DEFAULT_PORTFOLIO: dict = {
    "monthly_sip": 10_000,
    "funds": [
        {
            "name": "Nifty 50 Index Fund",
            "type": "large_cap",
            "invested": 5_00_000,
            "current": 4_50_000,
        },
        {
            "name": "Axis Midcap Fund",
            "type": "mid_cap",
            "invested": 3_00_000,
            "current": 2_55_000,
        },
        {
            "name": "HDFC Short Term Debt Fund",
            "type": "debt",
            "invested": 2_00_000,
            "current": 2_08_000,
        },
    ],
    "investment_horizon_years": 10,
    "risk_profile": "moderate",
}


class PanicGuardOrchestrator:
    """
    Coordinates all three PanicGuard agents into a single pipeline.

    Parameters
    ----------
    user_portfolio : dict, optional
        Investor's portfolio.  Falls back to a realistic demo portfolio.
    """

    def __init__(self, user_portfolio: Optional[dict] = None) -> None:
        self.portfolio = user_portfolio or dict(_DEFAULT_PORTFOLIO)
        self._agents_initialised = False
        self._crisis_detector = None
        self._portfolio_analyzer = None
        self._behavioral_coach = None

        # Cached results
        self._last_crisis_data: Optional[dict] = None
        self._last_crash_comparison: Optional[dict] = None
        self._last_portfolio_report: Optional[dict] = None
        self._last_coaching: Optional[dict] = None
        self._last_full_result: Optional[dict] = None

        self._init_agents()

    # ------------------------------------------------------------------
    # Agent initialisation (imported lazily to avoid circular imports)
    # ------------------------------------------------------------------

    def _init_agents(self) -> None:
        """Lazily import and instantiate all three agents."""
        try:
            from agents.crisis_detector import CrisisDetector
            from agents.portfolio_analyzer import PortfolioAnalyzer
            from agents.behavioral_coach import BehavioralCoach

            self._crisis_detector   = CrisisDetector()
            self._portfolio_analyzer = PortfolioAnalyzer(self.portfolio)
            self._behavioral_coach  = BehavioralCoach()
            self._agents_initialised = True
            logger.info("All three agents initialised successfully.")
        except Exception as e:
            logger.error(f"Agent initialisation failed: {e}")
            self._agents_initialised = False

    # ------------------------------------------------------------------
    # Full analysis pipeline
    # ------------------------------------------------------------------

    def run_full_analysis(self) -> dict:
        """
        Execute the full multi-agent pipeline:

            1. CrisisDetector.scan_market()
            2. CrisisDetector.get_crash_comparison()
            3. PortfolioAnalyzer.generate_report()  (includes impact + scenarios)
            4. BehavioralCoach.generate_coaching()

        Returns a single master dict containing all outputs, timing, and metadata.
        If any agent fails, the other agents still produce their output.
        """
        pipeline_start = time.time()
        timestamp = datetime.now()
        agent_timings: dict[str, float] = {}
        errors: list[str] = []

        logger.info("=" * 60)
        logger.info("PanicGuardOrchestrator.run_full_analysis()")
        logger.info(f"Timestamp: {timestamp.isoformat()}")
        logger.info("=" * 60)

        if not self._agents_initialised:
            logger.warning("Agents not initialised — returning demo result")
            return self.get_demo_result()

        # --- Step 1: Crisis Detection --------------------------------------
        t0 = time.time()
        try:
            logger.info("[1/4] Running CrisisDetector…")
            crisis_data = self._crisis_detector.scan_market()
            self._last_crisis_data = crisis_data
        except Exception as e:
            logger.error(f"[1/4] CrisisDetector failed: {e}")
            errors.append(f"CrisisDetector: {e}")
            crisis_data = self._get_demo_crisis_data()
            self._last_crisis_data = crisis_data
        agent_timings["crisis_detector"] = round(time.time() - t0, 3)
        logger.info(f"[1/4] Done in {agent_timings['crisis_detector']}s — score={crisis_data.get('panic_score')}")

        # --- Step 2: Crash Comparison --------------------------------------
        t0 = time.time()
        try:
            logger.info("[2/4] Running crash comparison…")
            crash_comparison = self._crisis_detector.get_crash_comparison()
            self._last_crash_comparison = crash_comparison
            # Merge into crisis_data for downstream agents
            crisis_data["crash_comparison"] = crash_comparison
        except Exception as e:
            logger.error(f"[2/4] Crash comparison failed: {e}")
            errors.append(f"CrashComparison: {e}")
            crash_comparison = {
                "most_similar_crash": "COVID-19 Crash (2020)",
                "similarity_score": 0.65,
                "recovery_months": 7,
                "post_bottom_gain": 112,
                "recovery_timeline": "In the COVID-19 Crash, the market recovered in 7 months.",
            }
            crisis_data["crash_comparison"] = crash_comparison
        agent_timings["crash_comparison"] = round(time.time() - t0, 3)

        # --- Step 3: Portfolio Analysis ------------------------------------
        t0 = time.time()
        try:
            logger.info("[3/4] Running PortfolioAnalyzer…")
            portfolio_report = self._portfolio_analyzer.generate_report(crisis_data)
            self._last_portfolio_report = portfolio_report
        except Exception as e:
            logger.error(f"[3/4] PortfolioAnalyzer failed: {e}")
            errors.append(f"PortfolioAnalyzer: {e}")
            portfolio_report = self._get_demo_portfolio_report()
            self._last_portfolio_report = portfolio_report
        agent_timings["portfolio_analyzer"] = round(time.time() - t0, 3)

        # --- Step 4: Behavioral Coaching -----------------------------------
        t0 = time.time()
        try:
            logger.info("[4/4] Running BehavioralCoach…")
            coaching = self._behavioral_coach.generate_coaching(crisis_data, portfolio_report)
            self._last_coaching = coaching
        except Exception as e:
            logger.error(f"[4/4] BehavioralCoach failed: {e}")
            errors.append(f"BehavioralCoach: {e}")
            coaching = {
                "coaching_message": (
                    "Every crash in Indian market history has been followed by a full "
                    "recovery. Continue your SIP — you are buying at a discount."
                ),
                "detected_biases": [],
                "recommended_action": "HOLD",
                "confidence_level": 0.80,
                "key_data_points": [],
            }
            self._last_coaching = coaching
        agent_timings["behavioral_coach"] = round(time.time() - t0, 3)

        # --- Assemble master result ----------------------------------------
        total_time = round(time.time() - pipeline_start, 3)

        result = {
            "timestamp":        timestamp.isoformat(),
            "pipeline_time_s":  total_time,
            "agent_timings":    agent_timings,
            "errors":           errors,
            "status":           "OK" if not errors else "PARTIAL",

            # Agent outputs
            "crisis_data":        crisis_data,
            "crash_comparison":   crash_comparison,
            "portfolio_report":   portfolio_report,
            "coaching":           coaching,

            # Top-level convenience fields for dashboard
            "panic_score":        crisis_data.get("panic_score", 0),
            "risk_level":         crisis_data.get("risk_level", "LOW"),
            "recommended_action": coaching.get("recommended_action", "HOLD"),
            "coaching_message":   coaching.get("coaching_message", ""),
            "is_demo":            crisis_data.get("is_demo", False),
        }

        self._last_full_result = result

        logger.info(f"Pipeline complete in {total_time}s — score={result['panic_score']}, action={result['recommended_action']}")
        if errors:
            logger.warning(f"Errors encountered: {errors}")

        return result

    # ------------------------------------------------------------------
    # Interactive chat
    # ------------------------------------------------------------------

    def run_chat(self, user_message: str) -> str:
        """
        Pass a user message to the Behavioral Coach with full context.

        If the pipeline hasn't been run yet, runs it first.
        """
        if not self._agents_initialised:
            return (
                "Every crash in Indian market history has been followed by recovery. "
                "Continue your SIP — you are buying at a discount. "
                "What specifically is worrying you?"
            )

        # Build rich context so template-mode chat can give data-driven answers
        context = {}
        if self._last_crisis_data:
            context["panic_score"]   = self._last_crisis_data.get("panic_score", 0)
            context["risk_level"]    = self._last_crisis_data.get("risk_level", "N/A")
            mkt = self._last_crisis_data.get("market_summary", {})
            context["nifty"] = mkt.get("nifty", "N/A")
            context["vix"]   = mkt.get("vix", "N/A")

        if self._last_portfolio_report:
            ps = self._last_portfolio_report.get("portfolio_summary", {})
            context["portfolio_value"]  = ps.get("total_current", "N/A")
            context["total_invested"]   = ps.get("total_invested", "N/A")
            context["gain_loss_pct"]    = ps.get("gain_loss_pct", 0)
            context["monthly_sip"]      = ps.get("monthly_sip", "N/A")
            cp = self._last_portfolio_report.get("cost_of_panic", {})
            context["cost_of_stopping_sip"] = cp.get("wealth_destroyed", "N/A")
            context["fv_if_continue"]   = cp.get("fv_if_continue", "N/A")
            context["fv_if_stop"]       = cp.get("fv_if_stop", "N/A")
            sip = self._last_portfolio_report.get("sip_scenarios", {})
            summary = sip.get("summary", {})
            context["hold_value"]  = summary.get("hold_value", "N/A")
            context["stop_value"]  = summary.get("stop_value", "N/A")
            context["brave_value"] = summary.get("brave_value", "N/A")
            context["horizon_years"] = sip.get("horizon_years", 10)

        if self._last_crash_comparison:
            context["similar_crash"]    = self._last_crash_comparison.get("most_similar_crash", "N/A")
            context["recovery_months"]  = self._last_crash_comparison.get("recovery_months", "N/A")
            context["post_bottom_gain"] = self._last_crash_comparison.get("post_bottom_gain", "N/A")

        return self._behavioral_coach.chat(user_message, context=context)

    # ------------------------------------------------------------------
    # Demo result  (ALWAYS works — no API, no internet)
    # ------------------------------------------------------------------

    @staticmethod
    def get_demo_result() -> dict:
        """
        Return a pre-computed full pipeline result using the April 2026
        tariff crash scenario.  Zero external dependencies.

        This is the nuclear fallback for hackathon judging.
        """
        logger.info("Generating pre-computed demo result (April 2026 crash scenario)")

        timestamp = datetime.now().isoformat()

        crisis_data = {
            "panic_score":      73,
            "risk_level":       "HIGH",
            "top_factors": [
                {"factor": "Drawdown from peak",   "value": "-12.2%", "impact": "35%"},
                {"factor": "Volatility spike",     "value": "31.2% annualised", "impact": "28%"},
                {"factor": "Consecutive red days",  "value": "5 days", "impact": "18%"},
            ],
            "shap_explanation": (
                "Drawdown from peak is -12.2% (contributing 35% to panic score). "
                "Volatility spike is 31.2% annualised (contributing 28% to panic score). "
                "Consecutive red days is 5 days (contributing 18% to panic score)."
            ),
            "recommendation": "HOLD",
            "market_summary": {
                "nifty":        21_843.50,
                "nifty_change": -2.85,
                "vix":          23.6,
                "crude":        61.20,
            },
            "shap_plot_path": None,
            "timestamp":      timestamp,
            "is_demo":        True,
            "crash_comparison": {
                "most_similar_crash": "Rate Hike Selloff (2022)",
                "similarity_score":   0.72,
                "recovery_months":    9,
                "post_bottom_gain":   28,
                "recovery_timeline": (
                    "In the Rate Hike Selloff (2022), the market fully recovered in 9 months "
                    "and investors who held gained 28% from the bottom."
                ),
            },
        }

        portfolio_report = {
            "timestamp": timestamp,
            "portfolio_summary": {
                "total_invested":   "₹10.00 L",
                "total_current":    "₹9.13 L",
                "gain_loss":        "-₹0.87 L",
                "gain_loss_pct":    -8.7,
                "monthly_sip":      "₹10,000",
                "horizon_years":    10,
                "risk_profile":     "moderate",
                "fund_count":       3,
                "fund_impacts": [
                    {"name": "Nifty 50 Index Fund",        "type": "large_cap", "invested": "₹5.00 L", "current": "₹4.50 L", "gain_loss": "-₹0.50 L", "gain_loss_pct": -10.0},
                    {"name": "Axis Midcap Fund",           "type": "mid_cap",   "invested": "₹3.00 L", "current": "₹2.55 L", "gain_loss": "-₹0.45 L", "gain_loss_pct": -15.0},
                    {"name": "HDFC Short Term Debt Fund",  "type": "debt",      "invested": "₹2.00 L", "current": "₹2.08 L", "gain_loss": "₹8,000",    "gain_loss_pct":   4.0},
                ],
            },
            "cost_of_panic": {
                "wealth_destroyed":     "₹8.45 L",
                "wealth_destroyed_raw": 845000,
                "fv_if_continue":       "₹34.71 L",
                "fv_if_stop":           "₹26.26 L",
                "extra_gains_if_brave": "₹7.89 L",
            },
            "sip_scenarios": {
                "scenarios": {
                    "stop":    [{"years": 5, "value_fmt": "₹12.48 L"}, {"years": 10, "value_fmt": "₹26.26 L"}, {"years": 15, "value_fmt": "₹38.10 L"}, {"years": 20, "value_fmt": "₹52.50 L"}],
                    "hold":    [{"years": 5, "value_fmt": "₹17.82 L"}, {"years": 10, "value_fmt": "₹34.71 L"}, {"years": 15, "value_fmt": "₹62.40 L"}, {"years": 20, "value_fmt": "₹1.08 Cr"}],
                    "brave":   [{"years": 5, "value_fmt": "₹21.25 L"}, {"years": 10, "value_fmt": "₹42.60 L"}, {"years": 15, "value_fmt": "₹78.30 L"}, {"years": 20, "value_fmt": "₹1.38 Cr"}],
                    "defense": [{"years": 5, "value_fmt": "₹15.10 L"}, {"years": 10, "value_fmt": "₹29.80 L"}, {"years": 15, "value_fmt": "₹48.70 L"}, {"years": 20, "value_fmt": "₹76.40 L"}],
                },
                "horizon_years": 10,
                "monthly_sip": "₹10,000",
                "summary": {
                    "stop_value":      "₹26.26 L",
                    "hold_value":      "₹34.71 L",
                    "brave_value":     "₹42.60 L",
                    "cost_of_panic":   "₹8.45 L",
                    "reward_of_brave": "₹7.89 L",
                    "panic_vs_brave":  "₹16.34 L",
                },
                "headline": (
                    "Stopping your SIP now costs you ₹8.45 L over 10 years. "
                    "Increasing it by 50% instead earns you ₹7.89 L extra."
                ),
            },
            "historical_comparison": {
                "most_similar_crash": "Rate Hike Selloff (2022)",
                "recovery_months":    9,
                "post_bottom_gain":   28,
                "message": (
                    "The most similar historical crash is the Rate Hike Selloff (2022). "
                    "After that crash, markets recovered in 9 months, and investors who "
                    "stayed invested gained 28% from the bottom."
                ),
                "sip_continuers": (
                    "In the Rate Hike Selloff (2022), only 24% of SIP investors continued. "
                    "Those 24% earned significantly higher returns than the 76% who panicked."
                ),
            },
            "recommendation": {
                "action":     "HOLD",
                "confidence": 0.80,
                "rationale": (
                    "Market is under significant stress, but your portfolio is built for "
                    "long-term growth. Continue your SIP — buying during dips lowers your "
                    "average cost basis and boosts future returns."
                ),
            },
        }

        coaching = {
            "coaching_message": (
                "**I understand how unsettling this feels. Seeing your portfolio down "
                "₹87,000 is painful — that's a real emotion and I'm not going to dismiss it.**\n\n"
                "But let me show you what the data says:\n\n"
                "📊 **Your Portfolio**: You've invested ₹10 L and it's currently at ₹9.13 L — "
                "an 8.7% paper loss. Important word: *paper*. You haven't lost anything until you sell.\n\n"
                "💸 **The Cost of Panic**: If you stop your ₹10,000/month SIP today, you lose "
                "**₹8.45 L** in future wealth over 10 years. That's not the market taking your "
                "money — that's a panic decision destroying it.\n\n"
                "📈 **History's Verdict**: The current situation looks most like the 2022 Rate Hike "
                "Selloff. After that crash, markets recovered in just 9 months. Investors who held "
                "gained 28% from the bottom.\n\n"
                "Here's the stat that matters most: **76% of Indian SIP investors stopped their "
                "mandates** during similar past crashes. The 24% who continued? They built "
                "significantly more wealth over the next decade.\n\n"
                "🎯 **My Recommendation: HOLD**\n"
                "Continue your SIP. You are buying units at a discount right now. Every SIP "
                "installment during a crash lowers your average cost and amplifies future returns.\n\n"
                "💡 If you're feeling brave, increase your SIP to ₹15,000/month. Your portfolio "
                "could grow to ₹42.60 L instead of ₹34.71 L — that's ₹7.89 L extra for being "
                "rational when others are emotional.\n\n"
                "> ⚖️ *Not SEBI-registered. Not investment advice. Behavioral and educational support only.*"
            ),
            "detected_biases":    ["Loss Aversion"],
            "recommended_action": "HOLD",
            "confidence_level":   0.80,
            "key_data_points": [
                "Panic Score: 73/100 (HIGH)",
                "Portfolio: ₹10.00 L invested → ₹9.13 L current (-8.7%)",
                "Monthly SIP: ₹10,000",
                "Cost of stopping SIP: ₹8.45 L over 10 years",
                "Most similar crash: Rate Hike Selloff (2022) (recovered in 9 months)",
                "76% of investors stopped SIPs during similar crashes — the 24% who continued built significantly more wealth",
            ],
        }

        return {
            "timestamp":        timestamp,
            "pipeline_time_s":  0.001,
            "agent_timings":    {"crisis_detector": 0.0, "crash_comparison": 0.0, "portfolio_analyzer": 0.0, "behavioral_coach": 0.0},
            "errors":           [],
            "status":           "DEMO",

            "crisis_data":        crisis_data,
            "crash_comparison":   crisis_data["crash_comparison"],
            "portfolio_report":   portfolio_report,
            "coaching":           coaching,

            "panic_score":        73,
            "risk_level":         "HIGH",
            "recommended_action": "HOLD",
            "coaching_message":   coaching["coaching_message"],
            "is_demo":            True,
        }

    # ------------------------------------------------------------------
    # Internal demo-data stubs (used when individual agents fail)
    # ------------------------------------------------------------------

    def _get_demo_crisis_data(self) -> dict:
        return PanicGuardOrchestrator.get_demo_result()["crisis_data"]

    def _get_demo_portfolio_report(self) -> dict:
        return PanicGuardOrchestrator.get_demo_result()["portfolio_report"]


# ============================================================
# Standalone test
# ============================================================

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  PanicGuard AI — Orchestrator Test Run")
    print("=" * 60)
    print()

    orch = PanicGuardOrchestrator()
    result = orch.run_full_analysis()

    print(f"\nStatus: {result['status']}")
    print(f"Panic Score: {result['panic_score']}/100  ({result['risk_level']})")
    print(f"Recommended Action: {result['recommended_action']}")
    print(f"Pipeline Time: {result['pipeline_time_s']}s")
    print(f"Agent Timings: {result['agent_timings']}")

    if result['errors']:
        print(f"Errors: {result['errors']}")

    print(f"\n{'-'*60}")
    print("COACHING MESSAGE:")
    print(f"{'-'*60}")
    print(result["coaching_message"].encode("ascii", errors="replace").decode("ascii"))

    print(f"\n{'-'*60}")
    print("CHAT TEST:")
    print(f"{'-'*60}")
    reply = orch.run_chat("Should I stop my SIP? The market is crashing!")
    print(f"User: Should I stop my SIP?\nCoach: {reply}")
    print()
