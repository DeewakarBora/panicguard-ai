"""
PanicGuard AI  —  Smoke Test
==============================
Quick validation that the entire pipeline works end-to-end.

Run:
    python -m tests.smoke_test           # from project root
    python tests/smoke_test.py           # direct
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure project root is importable
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _banner(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def _check(label: str, condition: bool) -> None:
    status = "PASS" if condition else "FAIL"
    symbol = "  +" if condition else "  X"  # ASCII safe for Windows
    print(f"  {symbol} {label}: {status}")
    if not condition:
        raise AssertionError(f"SMOKE TEST FAILED: {label}")


# ── Test 1: Demo result (zero-dependency) ──────────────────────────────

def test_demo_result():
    _banner("Test 1: get_demo_result()")
    from agents.orchestrator import PanicGuardOrchestrator

    t0 = time.time()
    result = PanicGuardOrchestrator.get_demo_result()
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.3f}s")

    _check("Returns dict", isinstance(result, dict))
    _check("Has status", result.get("status") == "DEMO")
    _check("Has panic_score", isinstance(result.get("panic_score"), (int, float)))
    _check("Has risk_level", result.get("risk_level") in ("LOW", "MEDIUM", "HIGH", "CRITICAL"))
    _check("Has coaching_message", len(result.get("coaching_message", "")) > 50)
    _check("Has crisis_data", isinstance(result.get("crisis_data"), dict))
    _check("Has portfolio_report", isinstance(result.get("portfolio_report"), dict))
    _check("Has coaching", isinstance(result.get("coaching"), dict))
    _check("Has crash_comparison", isinstance(result.get("crash_comparison"), dict))
    _check("Has agent_timings", isinstance(result.get("agent_timings"), dict))
    _check("Has timestamp", result.get("timestamp") is not None)
    _check("is_demo is True", result.get("is_demo") is True)

    # Check nested structures
    crisis = result["crisis_data"]
    _check("crisis.market_summary exists", isinstance(crisis.get("market_summary"), dict))
    _check("crisis.top_factors exists", isinstance(crisis.get("top_factors"), list))
    _check("crisis.shap_explanation exists", len(crisis.get("shap_explanation", "")) > 10)

    portfolio = result["portfolio_report"]
    _check("portfolio.portfolio_summary exists", isinstance(portfolio.get("portfolio_summary"), dict))
    _check("portfolio.cost_of_panic exists", isinstance(portfolio.get("cost_of_panic"), dict))
    _check("portfolio.sip_scenarios exists", isinstance(portfolio.get("sip_scenarios"), dict))

    coaching = result["coaching"]
    _check("coaching.detected_biases exists", isinstance(coaching.get("detected_biases"), list))
    _check("coaching.recommended_action exists", coaching.get("recommended_action") is not None)
    _check("coaching.key_data_points exists", len(coaching.get("key_data_points", [])) > 0)

    sip = portfolio["sip_scenarios"]
    _check("sip.scenarios has 4 keys", len(sip.get("scenarios", {})) == 4)
    _check("sip.summary exists", isinstance(sip.get("summary"), dict))

    print(f"\n  Demo result: score={result['panic_score']}, risk={result['risk_level']}")


# ── Test 2: Full pipeline (live or fallback) ───────────────────────────

def test_full_pipeline():
    _banner("Test 2: run_full_analysis() with demo portfolio")
    from agents.orchestrator import PanicGuardOrchestrator

    demo_portfolio = {
        "monthly_sip": 10_000,
        "funds": [
            {"name": "Test Large Cap", "type": "large_cap",
             "invested": 3_00_000, "current": 2_70_000},
            {"name": "Test Debt Fund", "type": "debt",
             "invested": 2_00_000, "current": 2_06_000},
        ],
        "investment_horizon_years": 10,
        "risk_profile": "moderate",
    }

    t0 = time.time()
    orch = PanicGuardOrchestrator(user_portfolio=demo_portfolio)
    result = orch.run_full_analysis()
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.3f}s")

    _check("Returns dict", isinstance(result, dict))
    _check("Status is OK or PARTIAL or DEMO",
           result.get("status") in ("OK", "PARTIAL", "DEMO"))
    _check("panic_score in 0-100",
           0 <= result.get("panic_score", -1) <= 100)
    _check("risk_level valid",
           result.get("risk_level") in ("LOW", "MEDIUM", "HIGH", "CRITICAL"))
    _check("coaching_message non-empty",
           len(result.get("coaching_message", "")) > 20)
    _check("pipeline_time_s is numeric",
           isinstance(result.get("pipeline_time_s"), (int, float)))
    _check("No uncaught exceptions", True)  # if we got here, no crash

    print(f"\n  Pipeline: status={result['status']}, score={result['panic_score']}, "
          f"time={result['pipeline_time_s']}s")

    if result.get("errors"):
        print(f"  Errors (gracefully handled): {result['errors']}")


# ── Test 3: Individual agents ──────────────────────────────────────────

def test_individual_agents():
    _banner("Test 3: Individual agent imports")

    # Crisis Detector
    try:
        from agents.crisis_detector import CrisisDetector
        cd = CrisisDetector()
        _check("CrisisDetector imports", True)
        _check("CrisisDetector has model or demo_mode",
               cd.model is not None or cd._demo_mode)
    except Exception as e:
        _check(f"CrisisDetector import ({e})", False)

    # Portfolio Analyzer
    try:
        from agents.portfolio_analyzer import PortfolioAnalyzer
        pa = PortfolioAnalyzer({
            "monthly_sip": 10000,
            "funds": [{"name": "Test", "type": "large_cap",
                       "invested": 100000, "current": 90000}],
            "investment_horizon_years": 10,
            "risk_profile": "moderate",
        })
        _check("PortfolioAnalyzer imports", True)
    except Exception as e:
        _check(f"PortfolioAnalyzer import ({e})", False)

    # Behavioral Coach
    try:
        from agents.behavioral_coach import BehavioralCoach
        bc = BehavioralCoach()
        _check("BehavioralCoach imports", True)
        _check("BehavioralCoach has provider chain",
               len(bc._provider_chain) >= 1)
    except Exception as e:
        _check(f"BehavioralCoach import ({e})", False)


# ── Test 4: Chat ───────────────────────────────────────────────────────

def test_chat():
    _banner("Test 4: Chat (run_chat)")
    from agents.orchestrator import PanicGuardOrchestrator

    orch = PanicGuardOrchestrator()
    reply = orch.run_chat("Should I stop my SIP?")
    _check("Chat returns string", isinstance(reply, str))
    _check("Chat reply non-empty", len(reply) > 20)
    print(f"  Reply preview: {reply[:100]}...")


# ── Test 5: Historical crashes data ────────────────────────────────────

def test_historical_data():
    _banner("Test 5: Historical crash data")
    from data.historical_crashes import (
        HISTORICAL_CRASHES, get_all_crashes,
        get_average_recovery_months, get_worst_drawdown,
    )
    _check("7 crashes loaded", len(HISTORICAL_CRASHES) == 7)
    _check("get_all_crashes works", len(get_all_crashes()) == 7)
    avg = get_average_recovery_months()
    _check(f"Avg recovery = {avg:.1f} months", avg > 0)
    worst = get_worst_drawdown()
    _check(f"Worst drawdown: {worst.name}", worst.nifty_peak_to_trough_pct < -30)


# ── Test 6: Helpers ────────────────────────────────────────────────────

def test_helpers():
    _banner("Test 6: Utility functions")
    from utils.helpers import format_inr

    _check("format_inr(100000) = lakhs", "L" in format_inr(100000))
    _check("format_inr(10000000) = crores", "Cr" in format_inr(10000000))
    _check("format_inr(0) safe", format_inr(0) is not None)
    _check("format_inr(-50000) safe", format_inr(-50000) is not None)


# ── Test 7: Edge cases ────────────────────────────────────────────────

def test_edge_cases():
    _banner("Test 7: Edge cases (zero/extreme values)")
    from agents.orchestrator import PanicGuardOrchestrator

    # Zero portfolio
    orch = PanicGuardOrchestrator(user_portfolio={
        "monthly_sip": 0,
        "funds": [],
        "investment_horizon_years": 1,
        "risk_profile": "conservative",
    })
    result = orch.run_full_analysis()
    _check("Zero portfolio doesn't crash",
           isinstance(result, dict) and result.get("status") is not None)

    # Huge portfolio
    orch2 = PanicGuardOrchestrator(user_portfolio={
        "monthly_sip": 1_00_000,
        "funds": [{"name": "Big Fund", "type": "large_cap",
                   "invested": 5_00_00_000, "current": 4_00_00_000}],
        "investment_horizon_years": 30,
        "risk_profile": "aggressive",
    })
    result2 = orch2.run_full_analysis()
    _check("Huge portfolio doesn't crash",
           isinstance(result2, dict) and result2.get("status") is not None)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                               ║
# ╚══════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  PanicGuard AI  --  Smoke Test Suite")
    print("=" * 60)

    tests = [
        test_demo_result,
        test_full_pipeline,
        test_individual_agents,
        test_chat,
        test_historical_data,
        test_helpers,
        test_edge_cases,
    ]

    passed = 0
    failed = 0
    t_start = time.time()

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"\n  !! {e}")
            failed += 1
        except Exception as e:
            print(f"\n  !! UNEXPECTED ERROR in {test_fn.__name__}: {e}")
            failed += 1

    total_time = time.time() - t_start

    print()
    print("=" * 60)
    if failed == 0:
        print(f"  + All {passed} smoke tests passed in {total_time:.2f}s")
    else:
        print(f"  RESULT: {passed} passed, {failed} FAILED in {total_time:.2f}s")
    print("=" * 60)
    print()

    sys.exit(1 if failed else 0)
