"""
Data Module: Fetch SIP / Mutual Fund Data
==========================================
Uses mftool to retrieve NAV history and scheme details for
Indian mutual funds.
"""

import pandas as pd
from mftool import Mftool
from typing import Optional

from utils.helpers import get_logger, retry

logger = get_logger(__name__)

mf = Mftool()


# ---------------------------------------------------------------------------
# Scheme Search
# ---------------------------------------------------------------------------

def search_scheme(query: str) -> list[dict]:
    """
    Search for mutual fund schemes by name keyword.

    Args:
        query: Scheme name or keyword (e.g., "HDFC Mid Cap").

    Returns:
        List of matching schemes with scheme_code and scheme_name.
    """
    try:
        schemes = mf.get_scheme_codes()
        matches = [
            {"scheme_code": code, "scheme_name": name}
            for code, name in schemes.items()
            if query.lower() in name.lower()
        ]
        logger.info(f"Found {len(matches)} schemes matching '{query}'")
        return matches[:20]  # Cap results
    except Exception as e:
        logger.error(f"Scheme search failed: {e}")
        return []


# ---------------------------------------------------------------------------
# NAV History
# ---------------------------------------------------------------------------

@retry(max_attempts=3, delay_seconds=1.5)
def get_nav_history(scheme_code: str) -> pd.DataFrame:
    """
    Fetch full NAV history for a mutual fund scheme.

    Args:
        scheme_code: mftool scheme code (string of digits).

    Returns:
        DataFrame with columns: ['date', 'nav']
    """
    logger.info(f"Fetching NAV history for scheme: {scheme_code}")
    data = mf.get_scheme_historical_nav(scheme_code, as_Dataframe=True)
    if data is None or data.empty:
        raise ValueError(f"No NAV data for scheme code: {scheme_code}")

    data = data.reset_index()
    data.columns = ["date", "nav"]
    data["date"] = pd.to_datetime(data["date"], format="%d-%m-%Y", errors="coerce")
    data["nav"] = pd.to_numeric(data["nav"], errors="coerce")
    data = data.dropna().sort_values("date").reset_index(drop=True)
    return data


# ---------------------------------------------------------------------------
# NAV Snapshot (Latest)
# ---------------------------------------------------------------------------

def get_latest_nav(scheme_code: str) -> dict:
    """
    Get the latest NAV and scheme name for a given scheme code.

    Returns:
        Dict with scheme_name, latest_nav, and date.
    """
    try:
        details = mf.get_scheme_details(scheme_code)
        return {
            "scheme_name": details.get("scheme_name", "Unknown"),
            "latest_nav": float(details.get("nav", 0)),
            "date": details.get("date", "N/A"),
        }
    except Exception as e:
        logger.error(f"Failed to get NAV for {scheme_code}: {e}")
        return {"scheme_name": "Unknown", "latest_nav": None, "date": None}


# ---------------------------------------------------------------------------
# Crash-Period NAV Analysis
# ---------------------------------------------------------------------------

def compute_fund_drawdown_during_crash(
    scheme_code: str,
    crash_start: str,
    crash_end: str,
) -> dict:
    """
    Compute the drawdown experienced by a fund during a specific crash window.

    Args:
        scheme_code: mftool scheme code.
        crash_start: Start date string "YYYY-MM-DD".
        crash_end:   End date string "YYYY-MM-DD".

    Returns:
        Dict with peak_nav, trough_nav, drawdown_pct, recovery info.
    """
    nav_df = get_nav_history(scheme_code)
    mask = (nav_df["date"] >= crash_start) & (nav_df["date"] <= crash_end)
    window = nav_df[mask]

    if window.empty:
        return {"error": f"No NAV data found between {crash_start} and {crash_end}"}

    peak_nav   = window["nav"].max()
    trough_nav = window["nav"].min()
    drawdown_pct = (trough_nav - peak_nav) / peak_nav * 100

    # Recovery: check if NAV reached peak again after crash_end
    post_crash = nav_df[nav_df["date"] > crash_end]
    recovery_date = post_crash[post_crash["nav"] >= peak_nav]["date"].min()
    recovered = not pd.isna(recovery_date)

    return {
        "scheme_code": scheme_code,
        "crash_period": f"{crash_start} → {crash_end}",
        "peak_nav": round(float(peak_nav), 4),
        "trough_nav": round(float(trough_nav), 4),
        "drawdown_pct": round(float(drawdown_pct), 2),
        "recovered": recovered,
        "recovery_date": str(recovery_date.date()) if recovered else "Not yet recovered",
    }
