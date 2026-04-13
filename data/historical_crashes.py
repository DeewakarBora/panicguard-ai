"""
Data Module: Historical Crash Reference Data
============================================
Stores pre-curated data about major Indian market crash events.
Used by the Portfolio Analyzer for analogue matching and context.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CrashEvent:
    """Represents a single historical market crash event."""
    name: str
    start_date: str
    end_date: str
    trigger: str
    nifty_peak_to_trough_pct: float    # Negative float
    nifty_recovery_months: int
    sensex_peak_to_trough_pct: float   # Negative float
    typical_sip_continuers_outcome: str
    typical_panic_sellers_outcome: str
    key_lesson: str
    global_context: Optional[str] = None


# ============================================================
# Historical Crash Database
# ============================================================

HISTORICAL_CRASHES: list[CrashEvent] = [

    CrashEvent(
        name="Global Financial Crisis (2008–09)",
        start_date="2008-01-01",
        end_date="2009-03-31",
        trigger="US subprime mortgage collapse, Lehman Brothers bankruptcy.",
        nifty_peak_to_trough_pct=-60.9,
        nifty_recovery_months=24,
        sensex_peak_to_trough_pct=-61.0,
        typical_sip_continuers_outcome=(
            "Investors who continued SIPs bought units at decade-low NAVs. "
            "By 2012, their SIP portfolios had grown 3x from bottom valuations."
        ),
        typical_panic_sellers_outcome=(
            "Those who stopped SIPs and sold in early 2009 locked in 40–60% losses "
            "and missed the subsequent 190% recovery over 3 years."
        ),
        key_lesson="This was a once-in-a-generation buying opportunity for disciplined SIP investors.",
        global_context="Synchronized global bear market. Even legendary investors like Warren Buffett were buying.",
    ),

    CrashEvent(
        name="European Debt Crisis Selloff (2011)",
        start_date="2011-07-01",
        end_date="2011-12-31",
        trigger="Greece sovereign debt crisis, Euro zone contagion fears.",
        nifty_peak_to_trough_pct=-28.0,
        nifty_recovery_months=14,
        sensex_peak_to_trough_pct=-28.5,
        typical_sip_continuers_outcome=(
            "Investors who held through recovered fully by early 2013 and gained 35%+ by Bullrun of 2014."
        ),
        typical_panic_sellers_outcome=(
            "Panic sellers who exited at the bottom missed a 45% recovery rally over the next 18 months."
        ),
        key_lesson="India-specific fundamentals were sound. External shock, not structural failure.",
    ),

    CrashEvent(
        name="China Slowdown & Global Selloff (2015–16)",
        start_date="2015-08-01",
        end_date="2016-02-29",
        trigger="China currency devaluation, commodity price collapse, yuan fears.",
        nifty_peak_to_trough_pct=-22.0,
        nifty_recovery_months=9,
        sensex_peak_to_trough_pct=-21.5,
        typical_sip_continuers_outcome=(
            "SIP continuers saw portfolios recover and return to new highs by mid-2016."
        ),
        typical_panic_sellers_outcome=(
            "Re-entry after the crash was at higher prices — sellers missed the recovery."
        ),
        key_lesson="Short and sharp corrections are the norm, not the exception.",
    ),

    CrashEvent(
        name="IL&FS Crisis & NBFC Contagion (2018–19)",
        start_date="2018-09-01",
        end_date="2019-03-31",
        trigger="IL&FS default, NBFC liquidity crisis, FII outflows.",
        nifty_peak_to_trough_pct=-16.0,
        nifty_recovery_months=8,
        sensex_peak_to_trough_pct=-16.5,
        typical_sip_continuers_outcome=(
            "Smallcap/midcap SIP investors suffered more, but large-cap SIPs recovered fully by Q3 2019."
        ),
        typical_panic_sellers_outcome=(
            "Many retail investors exited mid/smallcap funds at the worst time, locking in permanent losses."
        ),
        key_lesson="Sector-specific crises feel systemic but rarely are. Diversification is your shield.",
    ),

    CrashEvent(
        name="COVID-19 Crash (2020)",
        start_date="2020-02-15",
        end_date="2020-03-24",
        trigger="Global pandemic declaration, nationwide lockdowns, economic shutdown fears.",
        nifty_peak_to_trough_pct=-38.4,
        nifty_recovery_months=7,
        sensex_peak_to_trough_pct=-38.7,
        typical_sip_continuers_outcome=(
            "SIP investors who continued through March–April 2020 saw their portfolios double by Dec 2020. "
            "This was the greatest wealth creation event of the decade for disciplined investors."
        ),
        typical_panic_sellers_outcome=(
            "Investors who stopped SIPs in March 2020 and re-entered in Dec 2020 "
            "paid 85% higher prices for the same units."
        ),
        key_lesson=(
            "The fastest 38% crash in Indian market history was followed by the fastest 85% recovery. "
            "Patience of 7 months created generational wealth."
        ),
        global_context="Synchronised global shock. All asset classes fell simultaneously except gold.",
    ),

    CrashEvent(
        name="Global Rate Hike Selloff (2022)",
        start_date="2022-01-01",
        end_date="2022-06-30",
        trigger="US Fed aggressive rate hikes (75bps), global inflation shock, tech selloff.",
        nifty_peak_to_trough_pct=-16.0,
        nifty_recovery_months=9,
        sensex_peak_to_trough_pct=-15.5,
        typical_sip_continuers_outcome=(
            "Nifty fully recovered and reached new all-time highs by Q4 2022 for those who stayed."
        ),
        typical_panic_sellers_outcome=(
            "Those who stopped SIPs missed the second-half 2022 and 2023 bull run."
        ),
        key_lesson="India's macro fundamentals diverged positively from global peers during this period.",
    ),

    CrashEvent(
        name="April 2026 Correction",
        start_date="2026-04-01",
        end_date="2026-04-30",
        trigger="Geopolitical tensions, global risk-off, FII outflows, domestic earnings pressure.",
        nifty_peak_to_trough_pct=-12.0,   # Preliminary estimate
        nifty_recovery_months=0,           # Ongoing — TBD
        sensex_peak_to_trough_pct=-12.5,
        typical_sip_continuers_outcome=(
            "TBD — we are in the middle of this event. Historical base rates strongly favour continuers."
        ),
        typical_panic_sellers_outcome=(
            "76% of SIP mandates paused. These investors are buying back at higher prices "
            "if history repeats — as it has in every prior instance."
        ),
        key_lesson=(
            "You are living through this crash right now. "
            "Every investor who has held through a crash of this magnitude has been rewarded within 12 months."
        ),
        global_context="Current event. No hindsight available — but all prior analogues point to recovery.",
    ),
]


# ============================================================
# Lookup Functions
# ============================================================

def get_all_crashes() -> list[CrashEvent]:
    """Return the complete crash history database."""
    return HISTORICAL_CRASHES


def get_crash_by_name(name: str) -> Optional[CrashEvent]:
    """Lookup a crash event by name (case-insensitive partial match)."""
    name_lower = name.lower()
    for crash in HISTORICAL_CRASHES:
        if name_lower in crash.name.lower():
            return crash
    return None


def get_average_recovery_months() -> float:
    """Compute average recovery time across all historical crashes."""
    completed = [c for c in HISTORICAL_CRASHES if c.nifty_recovery_months > 0]
    if not completed:
        return 0.0
    return sum(c.nifty_recovery_months for c in completed) / len(completed)


def get_worst_drawdown() -> CrashEvent:
    """Return the crash event with the deepest Nifty drawdown."""
    return min(HISTORICAL_CRASHES, key=lambda c: c.nifty_peak_to_trough_pct)
