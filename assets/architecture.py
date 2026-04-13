"""
PanicGuard AI — Architecture Diagram Generator
================================================
Generates a publication-quality architecture diagram showing the multi-agent
pipeline from data sources → agents → orchestrator → dashboard.

Run:
    python assets/architecture.py
Outputs:
    assets/architecture.png  (1920×1080, presentaton quality)
    assets/architecture.svg  (vector, scalable)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Color Palette ─────────────────────────────────────────────────────
BG          = "#0f0c29"
CARD_BG     = "#1a1033"
CARD_BORDER = "#8b5cf6"
PURPLE      = "#8b5cf6"
PURPLE_DARK = "#6d28d9"
BLUE        = "#3b82f6"
GREEN       = "#22c55e"
ORANGE      = "#f97316"
AMBER       = "#f59e0b"
TEXT_MAIN   = "#f0f0f0"
TEXT_DIM    = "#94a3b8"
TEXT_BADGE  = "#c4b5fd"
ARROW_CLR   = "#6d28d9"


def _card(ax, x, y, w, h, title, subtitle="", badge="",
          color=CARD_BG, border=CARD_BORDER, title_color=TEXT_MAIN,
          badge_color=PURPLE):
    """Draw a glassmorphism-style rounded card."""
    # Outer glow
    glow = FancyBboxPatch((x - 0.012, y - 0.012), w + 0.024, h + 0.024,
                           boxstyle="round,pad=0.01",
                           facecolor="none",
                           edgecolor=border,
                           linewidth=1.2,
                           alpha=0.35,
                           zorder=2)
    ax.add_patch(glow)

    # Main card
    card = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.01",
                           facecolor=color,
                           edgecolor=border,
                           linewidth=1.8,
                           alpha=0.95,
                           zorder=3)
    ax.add_patch(card)

    cx = x + w / 2
    cy = y + h / 2

    # Badge (top)
    if badge:
        ax.text(cx, y + h + 0.012, badge,
                ha="center", va="bottom",
                fontsize=7.5, color=badge_color, fontweight="bold",
                fontfamily="monospace", zorder=5)

    # Title
    ax.text(cx, cy + (0.022 if subtitle else 0), title,
            ha="center", va="center",
            fontsize=11, color=title_color, fontweight="bold",
            zorder=5)

    # Subtitle
    if subtitle:
        ax.text(cx, cy - 0.028, subtitle,
                ha="center", va="center",
                fontsize=8, color=TEXT_DIM,
                zorder=5)


def _arrow(ax, x1, y1, x2, y2, label=""):
    """Draw a gradient-style arrow with optional label."""
    ax.annotate("",
                xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=ARROW_CLR,
                    lw=2.0,
                    connectionstyle="arc3,rad=0.0",
                ),
                zorder=4)
    if label:
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        ax.text(mx + 0.015, my, label,
                ha="left", va="center",
                fontsize=7, color=TEXT_DIM, fontstyle="italic", zorder=5)


def _horiz_arrow(ax, x1, y, x2, label=""):
    _arrow(ax, x1, y, x2, y, label)


def _vert_arrow(ax, x, y1, y2, label=""):
    _arrow(ax, x, y1, x, y2, label)


def _section_label(ax, x, y, text):
    ax.text(x, y, text,
            ha="left", va="center",
            fontsize=8.5, color=TEXT_DIM,
            fontweight="600",
            fontfamily="monospace",
            alpha=0.7,
            zorder=5)


def _divider(ax, y):
    ax.axhline(y, color=CARD_BORDER, linewidth=0.6, alpha=0.2, zorder=1)


def generate():
    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ── Background gradient bands ─────────────────────────────────────
    for i in range(80):
        alpha = 0.012 - i * 0.00012
        ax.axhspan(i / 80, (i + 1) / 80,
                   facecolor="#1a1145", alpha=max(0, alpha), zorder=0)

    # ── Title ─────────────────────────────────────────────────────────
    ax.text(0.5, 0.955,
            "🛡️  PanicGuard AI  —  Multi-Agent Architecture",
            ha="center", va="center",
            fontsize=22, color=TEXT_MAIN, fontweight="900",
            zorder=6)
    ax.text(0.5, 0.918,
            "Autonomous pipeline: Market Detection  →  Portfolio Impact  →  Behavioral Coaching",
            ha="center", va="center",
            fontsize=11, color=TEXT_DIM, zorder=6)

    # ══════════════════════════════════════════════════════════════════
    # LAYER 1 — Data Sources
    # ══════════════════════════════════════════════════════════════════
    _section_label(ax, 0.022, 0.848, "LAYER 1 — DATA SOURCES")
    _divider(ax, 0.835)

    ds_y = 0.760
    ds_h = 0.072
    ds_w = 0.20

    _card(ax, 0.12,  ds_y, ds_w, ds_h,
          "Yahoo Finance",
          "Nifty 50 · India VIX · Crude Oil",
          badge="yfinance API",
          border="#3b82f6", badge_color="#93c5fd")

    _card(ax, 0.40,  ds_y, ds_w, ds_h,
          "AMFI / mftool",
          "Mutual Fund NAV · SIP Data",
          badge="mftool API",
          border="#22c55e", badge_color="#86efac")

    _card(ax, 0.68,  ds_y, ds_w, ds_h,
          "Historical Crash DB",
          "7 Indian market crashes · Recovery data",
          badge="BSE / NSE Archives",
          border=AMBER, badge_color="#fde68a")

    # ══════════════════════════════════════════════════════════════════
    # ARROWS DS → Agents
    # ══════════════════════════════════════════════════════════════════
    _vert_arrow(ax, 0.22,  ds_y, 0.664)
    _vert_arrow(ax, 0.50,  ds_y, 0.664)
    _vert_arrow(ax, 0.78,  ds_y, 0.664)

    # ══════════════════════════════════════════════════════════════════
    # LAYER 2 — Agents
    # ══════════════════════════════════════════════════════════════════
    _section_label(ax, 0.022, 0.718, "LAYER 2 — AUTONOMOUS AGENTS")
    _divider(ax, 0.706)

    ag_y = 0.540
    ag_h = 0.122
    ag_w = 0.22

    # Agent 1
    _card(ax, 0.09, ag_y, ag_w, ag_h,
          "Agent 1: Crisis Detector",
          "XGBoost · SHAP · 15 Features\nPanic Score 0–100 · Risk Level",
          badge="ML-POWERED",
          border=ORANGE, title_color="#fed7aa", badge_color="#fdba74")

    # Agent 2
    _card(ax, 0.39, ag_y, ag_w, ag_h,
          "Agent 2: Portfolio Analyzer",
          "SIP Scenario Engine · ₹ Projections\n4 Scenarios · 5/10/15/20yr Horizons",
          badge="FINANCIAL MATH",
          border=AMBER, title_color="#fef08a", badge_color="#fde68a")

    # Agent 3
    _card(ax, 0.69, ag_y, ag_w, ag_h,
          "Agent 3: Behavioral Coach",
          "LLM (Claude / GPT-4o) · Template Fallback\nBias Detection · Conversational Chat",
          badge="BEHAVIORAL AI",
          border=GREEN, title_color="#bbf7d0", badge_color="#86efac")

    # Horizontal flow arrows between agents
    _horiz_arrow(ax, 0.31, ag_y + ag_h / 2, 0.39, label="crisis_data")
    _horiz_arrow(ax, 0.61, ag_y + ag_h / 2, 0.69, label="portfolio_report")

    # ── Mini badge boxes inside agent cards ──────────────────────────
    def _mini_badge(ax, x, y, text, color):
        ax.text(x, y, text, ha="center", va="center",
                fontsize=6.5, color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor=color + "22",
                          edgecolor=color + "66",
                          linewidth=0.8),
                zorder=6)

    _mini_badge(ax, 0.200, ag_y + 0.019, "XGBoost", ORANGE)
    _mini_badge(ax, 0.155, ag_y + 0.019, "SHAP", "#f97316")
    _mini_badge(ax, 0.498, ag_y + 0.019, "SMOTE", AMBER)
    _mini_badge(ax, 0.455, ag_y + 0.019, "CAGR", AMBER)
    _mini_badge(ax, 0.798, ag_y + 0.019, "Claude", GREEN)
    _mini_badge(ax, 0.757, ag_y + 0.019, "GPT-4o", GREEN)

    # ══════════════════════════════════════════════════════════════════
    # ARROWS Agents → Orchestrator
    # ══════════════════════════════════════════════════════════════════
    _vert_arrow(ax, 0.20,  ag_y,  0.428)
    _vert_arrow(ax, 0.50,  ag_y,  0.428)
    _vert_arrow(ax, 0.80,  ag_y,  0.428)

    # ══════════════════════════════════════════════════════════════════
    # LAYER 3 — Orchestrator
    # ══════════════════════════════════════════════════════════════════
    _section_label(ax, 0.022, 0.490, "LAYER 3 — ORCHESTRATOR")
    _divider(ax, 0.478)

    oc_x, oc_y, oc_w, oc_h = 0.22, 0.330, 0.56, 0.092
    _card(ax, oc_x, oc_y, oc_w, oc_h,
          "PanicGuard Orchestrator",
          "Autonomous pipeline  ·  Per-agent error isolation  ·  Timed execution  ·  Demo fallback",
          badge="PanicGuardOrchestrator.run_full_analysis()",
          border=PURPLE, title_color="#c4b5fd", badge_color="#a78bfa")

    # Timing badges
    for label, xi in [("1.4s typical", 0.310), ("4 agents", 0.500), ("zero-crash", 0.655)]:
        _mini_badge(ax, xi, oc_y + 0.012, label, PURPLE)

    # ══════════════════════════════════════════════════════════════════
    # ARROW Orchestrator → Dashboard
    # ══════════════════════════════════════════════════════════════════
    _vert_arrow(ax, 0.50, oc_y, 0.240)

    # ══════════════════════════════════════════════════════════════════
    # LAYER 4 — Dashboard
    # ══════════════════════════════════════════════════════════════════
    _section_label(ax, 0.022, 0.290, "LAYER 4 — STREAMLIT DASHBOARD")
    _divider(ax, 0.278)

    db_x, db_y, db_w, db_h = 0.09, 0.080, 0.82, 0.150

    # Background card for the whole dashboard layer
    outer = FancyBboxPatch((db_x - 0.01, db_y - 0.012), db_w + 0.02, db_h + 0.024,
                            boxstyle="round,pad=0.01",
                            facecolor="#13103a", edgecolor=PURPLE, linewidth=1.4,
                            alpha=0.7, zorder=2)
    ax.add_patch(outer)

    ax.text(0.5, db_y + db_h + 0.010,
            "Streamlit Dashboard  ·  Dark Glassmorphism UI  ·  Works Offline",
            ha="center", va="center",
            fontsize=9, color=TEXT_DIM, zorder=5)

    # Feature tiles
    tiles = [
        ("🎯", "Panic Score Gauge",    "0–100 · Risk Level · Animated",    ORANGE),
        ("📊", "Market Snapshot",      "Nifty · VIX · Crude Oil",          BLUE),
        ("💰", "SIP Scenarios",        "4 paths · ₹ projections",          AMBER),
        ("🧠", "AI Coaching",          "Empathetic · Bias-aware · Chat",   GREEN),
        ("📜", "Historical Crashes",   "7 events · Recovery timeline",     "#ec4899"),
        ("🔬", "SHAP Explainability",  "Waterfall · Beeswarm · Plain Eng.",PURPLE),
    ]

    tile_w = 0.128
    tile_h = 0.100
    padding = 0.013
    total_w = 6 * tile_w + 5 * padding
    x_start = (1 - total_w) / 2

    for i, (emoji, title, sub, color) in enumerate(tiles):
        tx = x_start + i * (tile_w + padding)
        ty = db_y + 0.015

        tile = FancyBboxPatch((tx, ty), tile_w, tile_h,
                               boxstyle="round,pad=0.008",
                               facecolor=color + "14",
                               edgecolor=color + "55",
                               linewidth=1.2, alpha=1.0, zorder=4)
        ax.add_patch(tile)

        cx = tx + tile_w / 2
        ax.text(cx, ty + tile_h - 0.018, emoji,
                ha="center", va="center", fontsize=16, zorder=5)
        ax.text(cx, ty + tile_h * 0.42, title,
                ha="center", va="center",
                fontsize=8.0, color=TEXT_MAIN, fontweight="bold", zorder=5)
        ax.text(cx, ty + tile_h * 0.16, sub,
                ha="center", va="center",
                fontsize=6.5, color=TEXT_DIM, zorder=5)

    # ── Footer ────────────────────────────────────────────────────────
    ax.text(0.5, 0.022,
            "AI Automate 2026 Hackathon  ·  XGBoost + SHAP + Streamlit + Claude/GPT-4o  ·  github.com/panicguard-ai",
            ha="center", va="center",
            fontsize=8, color=TEXT_DIM, alpha=0.7, zorder=5)

    # ── Save ─────────────────────────────────────────────────────────
    out_dir = Path(__file__).resolve().parent
    png_path = out_dir / "architecture.png"
    svg_path = out_dir / "architecture.svg"

    plt.tight_layout(pad=0)
    plt.savefig(str(png_path), dpi=100, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.savefig(str(svg_path), format="svg", bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close()

    print(f"  + Saved: {png_path}")
    print(f"  + Saved: {svg_path}")
    return str(png_path)


if __name__ == "__main__":
    print("\nGenerating PanicGuard AI architecture diagram...")
    path = generate()
    print(f"\nDone!  Open: {path}\n")
