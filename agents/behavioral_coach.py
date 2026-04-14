"""
Agent 3: Behavioral Coach  —  AI-Powered Crisis Counselor
==========================================================
Combines crisis data from Agent 1 and portfolio analysis from Agent 2
into a personalised, empathetic coaching message.  Powered by Claude or
GPT-4o, with a full template-based fallback when no API key is available
(critical for hackathon demo/judging).

Integrates with:
    - utils/config.py  (LLM keys, model names, system prompt, thresholds)
    - utils/helpers.py  (logging, format_inr)

Usage:
    coach    = BehavioralCoach()
    result   = coach.generate_coaching(crisis_data, portfolio_report)
    reply    = coach.chat("Should I stop my SIP?", context)
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    COACH_SYSTEM_PROMPT,
    LLM_MAX_TOKENS,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)
from utils.helpers import format_inr, get_logger

logger = get_logger("agents.behavioral_coach")


# ============================================================
# Bias detection keywords
# ============================================================

_BIAS_PATTERNS: dict[str, list[str]] = {
    "Loss Aversion": [
        "losing", "lost", "loss", "ruined", "wiped out", "destroyed",
        "can't afford to lose", "all my money",
    ],
    "Recency Bias": [
        "always falls", "never recovers", "it's different this time",
        "worst ever", "will keep falling", "no recovery",
    ],
    "Herd Mentality": [
        "everyone is selling", "all my friends", "people are",
        "everyone says", "the whole market", "news says",
    ],
    "Anchoring": [
        "bought at", "was at", "used to be", "it was",
        "peak price", "all-time high",
    ],
    "Availability Heuristic": [
        "saw on news", "read that", "they said",
        "youtube", "twitter", "social media", "whatsapp",
    ],
    "Panic / Emotional": [
        "sell everything", "get out", "stop sip", "cancel sip",
        "withdraw", "can't sleep", "scared", "afraid", "panic",
    ],
}


class BehavioralCoach:
    """
    LLM-powered behavioral finance coach.

    Supports Anthropic Claude and OpenAI GPT-4o (selected via config).
    Falls back to a rich template-based response when no API key is available.
    """

    def __init__(self) -> None:
        self.provider: str = LLM_PROVIDER.lower()
        self.conversation_history: list[dict] = []
        self._has_api_key = self._check_api_key()
        # Build ordered fallback chain: primary → secondary → template
        self._provider_chain = self._build_provider_chain()
        logger.info(
            f"BehavioralCoach init — provider={self.provider}, "
            f"api_key={'present' if self._has_api_key else 'MISSING (template mode)'}, "
            f"chain={[p for p in self._provider_chain]}"
        )

    # ------------------------------------------------------------------
    # API key validation & fallback chain
    # ------------------------------------------------------------------

    def _check_api_key(self) -> bool:
        if self.provider == "anthropic" and ANTHROPIC_API_KEY:
            return True
        if self.provider == "openai" and OPENAI_API_KEY:
            return True
        return False

    def _build_provider_chain(self) -> list[str]:
        """Build ordered list of available LLM providers."""
        chain = []
        if self.provider == "anthropic":
            if ANTHROPIC_API_KEY:
                chain.append("anthropic")
            if OPENAI_API_KEY:
                chain.append("openai")
        else:
            if OPENAI_API_KEY:
                chain.append("openai")
            if ANTHROPIC_API_KEY:
                chain.append("anthropic")
        chain.append("template")  # always available
        return chain

    # ------------------------------------------------------------------
    # Bias detection
    # ------------------------------------------------------------------

    def detect_biases(self, text: str) -> list[str]:
        """Detect behavioral biases from user text."""
        text_lower = text.lower()
        detected: list[str] = []
        for bias, keywords in _BIAS_PATTERNS.items():
            if any(kw in text_lower for kw in keywords):
                detected.append(bias)
        return detected

    # ------------------------------------------------------------------
    # Main coaching generation  (Agent 1 + Agent 2 → advice)
    # ------------------------------------------------------------------

    def generate_coaching(
        self,
        crisis_data: dict,
        portfolio_report: dict,
    ) -> dict:
        """
        Generate personalised coaching combining crisis and portfolio data.

        Parameters
        ----------
        crisis_data      : output of CrisisDetector.scan_market()
        portfolio_report : output of PortfolioAnalyzer.generate_report()

        Returns
        -------
        dict with keys: coaching_message, detected_biases, recommended_action,
                        confidence_level, key_data_points
        """
        logger.info("=" * 50)
        logger.info("BehavioralCoach.generate_coaching()")
        logger.info("=" * 50)

        # --- Extract key data points for coaching ---------------------------
        panic_score   = crisis_data.get("panic_score", 0)
        risk_level    = crisis_data.get("risk_level", "MEDIUM")
        explanation   = crisis_data.get("shap_explanation", "")
        mkt           = crisis_data.get("market_summary", {})
        recommendation = portfolio_report.get("recommendation", {})
        cost_section  = portfolio_report.get("cost_of_panic", {})
        sip_section   = portfolio_report.get("sip_scenarios", {})
        history       = portfolio_report.get("historical_comparison", {})
        portfolio_sum = portfolio_report.get("portfolio_summary", {})

        key_data_points = [
            f"Panic Score: {panic_score}/100 ({risk_level})",
            f"Portfolio: {portfolio_sum.get('total_invested', 'N/A')} invested → {portfolio_sum.get('total_current', 'N/A')} current ({portfolio_sum.get('gain_loss_pct', 0):+.1f}%)",
            f"Monthly SIP: {portfolio_sum.get('monthly_sip', 'N/A')}",
            f"Cost of stopping SIP: {cost_section.get('wealth_destroyed', 'N/A')} over {portfolio_sum.get('horizon_years', 10)} years",
            f"Most similar crash: {history.get('most_similar_crash', 'N/A')} (recovered in {history.get('recovery_months', '?')} months)",
            f"76% of investors stopped SIPs during similar crashes — the 24% who continued built significantly more wealth",
        ]

        # --- Build the context block for LLM --------------------------------
        context_block = self._build_llm_context(
            crisis_data, portfolio_report, key_data_points
        )

        # --- Generate response (LLM or template) ---------------------------
        if self._has_api_key:
            coaching_message = self._call_llm(context_block)
        else:
            coaching_message = self._template_response(
                panic_score, risk_level, portfolio_sum, cost_section,
                history, sip_section, recommendation
            )

        detected_biases = []
        if panic_score >= 60:
            detected_biases.append("Loss Aversion")
        if panic_score >= 80:
            detected_biases.append("Panic / Emotional")

        result = {
            "coaching_message":   coaching_message,
            "detected_biases":    detected_biases,
            "recommended_action": recommendation.get("action", "HOLD"),
            "confidence_level":   recommendation.get("confidence", 0.80),
            "key_data_points":    key_data_points,
        }

        logger.info(f"Coaching generated — action={result['recommended_action']}")
        return result

    # ------------------------------------------------------------------
    # Interactive chat
    # ------------------------------------------------------------------

    def chat(self, user_message: str, context: Optional[dict] = None) -> str:
        """
        Handle follow-up questions in a conversational manner.

        Parameters
        ----------
        user_message : the investor's question or concern.
        context      : optional dict with crisis + portfolio data.

        Returns
        -------
        Coach response as a string.
        """
        logger.info(f"Chat received: {user_message[:80]}")

        # Detect biases in user's message
        biases = self.detect_biases(user_message)
        if biases:
            logger.info(f"Biases detected in user message: {biases}")

        # Build context preamble on first message
        if context and len(self.conversation_history) == 0:
            preamble = self._build_llm_context_from_flat(context)
            enriched = f"{preamble}\n\n---\n\n**Investor says:** {user_message}"
        else:
            enriched = user_message

        self.conversation_history.append({"role": "user", "content": enriched})

        if self._has_api_key:
            reply = self._call_llm_chat(self.conversation_history)
        else:
            reply = self._template_chat_reply(user_message, biases, context)

        self.conversation_history.append({"role": "assistant", "content": reply})
        return reply

    def reset(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("Conversation history reset.")

    def get_history(self) -> list[dict]:
        return list(self.conversation_history)

    # ------------------------------------------------------------------
    # LLM context builders
    # ------------------------------------------------------------------

    def _build_llm_context(
        self,
        crisis_data: dict,
        portfolio_report: dict,
        key_data_points: list[str],
    ) -> str:
        """Build a rich prompt context for the coaching LLM call."""
        mkt         = crisis_data.get("market_summary", {})
        top_factors = crisis_data.get("top_factors", [])
        cost        = portfolio_report.get("cost_of_panic", {})
        scenarios   = portfolio_report.get("sip_scenarios", {})
        history     = portfolio_report.get("historical_comparison", {})
        rec         = portfolio_report.get("recommendation", {})

        factors_str = "\n".join(
            f"  - {f['factor']}: {f['value']} ({f['impact']} impact)"
            for f in top_factors
        )

        summary_block = scenarios.get("summary", {})

        return f"""## LIVE MARKET DATA
- Nifty 50: {mkt.get("nifty", "N/A")} ({mkt.get("nifty_change", 0):+.2f}% today)
- India VIX: {mkt.get("vix", "N/A")}
- Panic Score: {crisis_data.get("panic_score", 0)}/100 ({crisis_data.get("risk_level", "?")})

## WHAT'S DRIVING THE PANIC SCORE
{factors_str}

## INVESTOR'S PORTFOLIO
- Total invested: {portfolio_report.get("portfolio_summary", {}).get("total_invested", "N/A")}
- Current value: {portfolio_report.get("portfolio_summary", {}).get("total_current", "N/A")}
- Monthly SIP: {portfolio_report.get("portfolio_summary", {}).get("monthly_sip", "N/A")}
- Loss so far: {portfolio_report.get("portfolio_summary", {}).get("gain_loss", "N/A")} ({portfolio_report.get("portfolio_summary", {}).get("gain_loss_pct", 0):+.1f}%)

## COST OF PANIC-SELLING
- If continue SIP: portfolio grows to {cost.get("fv_if_continue", "N/A")}
- If stop SIP now: portfolio only reaches {cost.get("fv_if_stop", "N/A")}
- Wealth destroyed by panicking: {cost.get("wealth_destroyed", "N/A")}

## HISTORICAL COMPARISON
- {history.get("message", "N/A")}
- {history.get("sip_continuers", "")}

## SIP SCENARIOS ({scenarios.get("horizon_years", 10)}-year horizon)
- STOP SIP (panic): {summary_block.get("stop_value", "N/A")}
- CONTINUE SIP (hold): {summary_block.get("hold_value", "N/A")}
- INCREASE SIP 50% (brave): {summary_block.get("brave_value", "N/A")}

## KEY STAT
76% of Indian retail investors stopped their SIPs during similar past crashes.
The 24% who continued built significantly more wealth over the next 5-10 years.

## RECOMMENDATION FROM MODEL
Action: {rec.get("action", "HOLD")} (confidence: {rec.get("confidence", 0.8):.0%})
Rationale: {rec.get("rationale", "")}

---

Given all this data, provide empathetic, personalised coaching to help this
investor avoid a panic-driven decision. Acknowledge their fear. Show them
the data. Reference the most similar historical crash. Give a specific,
actionable recommendation. Identify any behavioral biases at play."""

    def _build_llm_context_from_flat(self, context: dict) -> str:
        """Build a context preamble from a flat key-value dict."""
        lines = ["## Current Context"]
        for k, v in context.items():
            lines.append(f"- **{k.replace('_', ' ').title()}**: {v}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------

    def _call_llm(self, context: str) -> str:
        """One-shot LLM call with full fallback chain: Claude → OpenAI → template."""
        messages = [{"role": "user", "content": context}]

        for provider in self._provider_chain:
            if provider == "template":
                break  # let caller decide template vs fallback
            try:
                if provider == "anthropic":
                    result = self._call_anthropic(messages)
                else:
                    result = self._call_openai(messages)
                if result and not result.startswith("I understand this market"):
                    return result  # success
            except Exception as e:
                logger.warning(f"{provider} failed in _call_llm: {e} — trying next")
                continue

        logger.info("All LLM providers exhausted — using fallback message")
        return self._fallback_message()

    def _call_llm_chat(self, messages: list[dict]) -> str:
        """Multi-turn chat call with full fallback chain."""
        for provider in self._provider_chain:
            if provider == "template":
                break
            try:
                if provider == "anthropic":
                    result = self._call_anthropic(messages)
                else:
                    result = self._call_openai(messages)
                if result and not result.startswith("I understand this market"):
                    return result
            except Exception as e:
                logger.warning(f"{provider} failed in _call_llm_chat: {e} — trying next")
                continue

        return self._fallback_message()

    def _call_anthropic(self, messages: list[dict]) -> str:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            response = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
                system=COACH_SYSTEM_PROMPT,
                messages=messages,
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return self._fallback_message()

    def _call_openai(self, messages: list[dict]) -> str:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            all_msgs = [{"role": "system", "content": COACH_SYSTEM_PROMPT}] + messages
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=all_msgs,
                max_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return self._fallback_message()

    # ------------------------------------------------------------------
    # Template-based fallback (NO API key required)
    # ------------------------------------------------------------------

    def _template_response(
        self,
        panic_score: int,
        risk_level: str,
        portfolio_sum: dict,
        cost_section: dict,
        history: dict,
        sip_section: dict,
        recommendation: dict,
    ) -> str:
        """
        Rich coaching message generated purely from data — no LLM needed.
        This is the hackathon demo fallback.
        """
        similar_crash = history.get("most_similar_crash", "COVID-19 Crash (2020)")
        recovery_mo   = history.get("recovery_months", 7)
        post_gain     = history.get("post_bottom_gain", 112)
        wealth_lost   = cost_section.get("wealth_destroyed", "N/A")
        action        = recommendation.get("action", "HOLD")
        summary       = sip_section.get("summary", {})

        if panic_score >= 80:
            empathy = (
                "I understand how terrifying this feels. Your portfolio is showing red "
                "across the board. The urge to sell everything and protect what's left "
                "is completely natural — it's your brain's fight-or-flight response. "
                "But right now, that instinct is your biggest financial enemy."
            )
        elif panic_score >= 60:
            empathy = (
                "Markets are clearly under stress, and it's normal to feel anxious "
                "when you see your hard-earned money decline. Your concern is valid. "
                "Let me show you what the data says about moments like this."
            )
        else:
            empathy = (
                "Markets are showing some turbulence, but this is well within "
                "the normal range of fluctuations. Let's look at your portfolio objectively."
            )

        return f"""**{empathy}**

Here's what the data tells us:

📊 **Your Portfolio Right Now**
Your portfolio is currently at {portfolio_sum.get('total_current', 'N/A')} — that's {portfolio_sum.get('gain_loss_pct', 0):+.1f}% from your invested amount of {portfolio_sum.get('total_invested', 'N/A')}. This paper loss has NOT been realised. You only lock in losses when you sell.

💸 **The Real Cost of Panicking**
If you stop your SIP of {portfolio_sum.get('monthly_sip', 'N/A')}/month today:
- **Continue SIP**: your portfolio grows to **{summary.get('hold_value', 'N/A')}** in {sip_section.get('horizon_years', 10)} years
- **Stop SIP (panic)**: it barely reaches **{summary.get('stop_value', 'N/A')}**
- **Wealth destroyed by panic**: **{wealth_lost}**

That's real money — your family's future — evaporated not by the market, but by a decision made in fear.

📈 **What History Teaches Us**
The current situation is most similar to the **{similar_crash}**.
- After that crash, the market fully recovered in just **{recovery_mo} months**
- Investors who stayed the course gained **{post_gain}%** from the bottom
- **76% of Indian SIP investors stopped their mandates** during that period — and they bought back at much higher prices later

The 24% who continued? They built significantly more wealth over the next decade.

🎯 **My Recommendation: {action}**
{recommendation.get('rationale', 'Continue your investment plan.')}

💡 **If you're feeling brave**, consider increasing your SIP by 50%. Historically, investors who increased SIPs during crashes earned extraordinary returns. Your portfolio could grow to **{summary.get('brave_value', 'N/A')}** — that's {summary.get('reward_of_brave', 'N/A')} more than the base case.

> ⚖️ *I am not SEBI-registered and this is not investment advice. This is behavioral and educational support based on data and historical patterns.*"""

    def _template_chat_reply(
        self,
        user_message: str,
        biases: list[str],
        context: Optional[dict],
    ) -> str:
        """
        Keyword-routed template replies — each branch injects real numbers
        from the analysis context so no two responses feel generic.
        """
        ctx = context or {}
        msg = user_message.lower()

        # ── Extract context values with sensible defaults ─────────────────
        panic_score     = ctx.get("panic_score", 73)
        risk_level      = ctx.get("risk_level", "HIGH")
        monthly_sip     = ctx.get("monthly_sip", "your SIP")
        portfolio_value = ctx.get("portfolio_value", "your portfolio")
        total_invested  = ctx.get("total_invested", "your invested amount")
        gain_loss_pct   = ctx.get("gain_loss_pct", 0)
        wealth_lost     = ctx.get("cost_of_stopping_sip", "lakhs of rupees")
        similar_crash   = ctx.get("similar_crash", "COVID-19 (2020)")
        recovery_months = ctx.get("recovery_months", 7)
        post_gain       = ctx.get("post_bottom_gain", 112)
        hold_value      = ctx.get("hold_value", "a much higher amount")
        stop_value      = ctx.get("stop_value", "a much lower amount")
        brave_value     = ctx.get("brave_value", "the highest potential outcome")
        horizon_years   = ctx.get("horizon_years", 10)
        vix             = ctx.get("vix", "N/A")

        # Format gain/loss as a signed percentage string
        try:
            gain_str = f" ({float(gain_loss_pct):+.1f}%)"
        except (TypeError, ValueError):
            gain_str = ""

        # Format VIX
        try:
            vix_str = f"{float(vix):.1f}"
            vix_high = float(vix) > 20
        except (TypeError, ValueError):
            vix_str = str(vix)
            vix_high = False

        # Format Nifty
        nifty = ctx.get("nifty", "N/A")
        try:
            nifty_str = f"{float(nifty):,.0f}"
        except (TypeError, ValueError):
            nifty_str = str(nifty)

        # ── Bias preamble (prepended when a strong bias is detected) ──────
        _bias_notes = {
            "Loss Aversion": (
                "🧠 **What I notice:** You're focused on the loss — that's **loss aversion**, "
                "where losses feel 2× more painful than equivalent gains. "
                "The loss you're seeing is on paper only. It becomes real only if you sell.\n\n"
            ),
            "Herd Mentality": (
                "🧠 **What I notice:** You may be reacting to what others are doing — "
                "classic **herd mentality**. The 76% who stopped SIPs during past crashes "
                "were the herd. The 24% who didn't are the ones who built wealth.\n\n"
            ),
            "Recency Bias": (
                "🧠 **What I notice:** You're assuming the current drop continues forever — "
                "that's **recency bias**. Every Indian crash in 20 years reversed. "
                "Average recovery: ~11 months.\n\n"
            ),
            "Panic / Emotional": (
                "🧠 **What I notice:** This feels urgent and scary — your amygdala is firing. "
                "That's a **panic response**, not a financial analysis. "
                "Permanent decisions made in peak fear almost always turn out to be costly.\n\n"
            ),
        }
        bias_preamble = ""
        for b in biases:
            if b in _bias_notes:
                bias_preamble = _bias_notes[b]
                break

        # ══════════════════════════════════════════════════════════════════
        # ROUTE 1 — SIP / stop / pause / cancel
        # ══════════════════════════════════════════════════════════════════
        if any(w in msg for w in ["sip", "stop", "pause", "cancel", "mandate"]):
            return (
                f"{bias_preamble}"
                f"**Don't stop your SIP — this is exactly the moment it works hardest for you.**\n\n"
                f"Your {monthly_sip}/month is currently buying units at a *discount*. "
                f"When the market falls, every rupee buys more units than it would in a rally. "
                f"That's rupee-cost averaging — the core advantage of a SIP over a lump sum.\n\n"
                f"📊 **The live numbers:** Panic score is **{panic_score}/100** ({risk_level}). "
                f"This is most similar to the **{similar_crash}**, which recovered in "
                f"**{recovery_months} months** and then gained **+{post_gain}%** from the bottom.\n\n"
                f"💸 **What stopping costs you over {horizon_years} years:**\n"
                f"- Continue SIP → **{hold_value}**\n"
                f"- Stop SIP now → only **{stop_value}**\n"
                f"- **Wealth destroyed: {wealth_lost}** — that's the price of this one decision.\n\n"
                f"76% of investors stopped SIPs during similar crashes. "
                f"The 24% who didn't built significantly more wealth. Which side do you want to be on?"
            )

        # ══════════════════════════════════════════════════════════════════
        # ROUTE 2 — scared / fear / worried / panic / anxious
        # (checked early — emotional state overrides generic market queries)
        # ══════════════════════════════════════════════════════════════════
        elif any(w in msg for w in ["scared", "fear", "worried", "worry", "panic", "anxious",
                                     "nervous", "stress", "tense", "afraid", "terrif"]):
            science_line = (
                bias_preamble if bias_preamble else
                "🧠 **The science:** Fear of loss is 2× stronger than pleasure of gain — "
                "your brain is wired to overreact to downturns. That wiring kept our "
                "ancestors alive, but it destroys investor wealth.\n\n"
            )
            return (
                f"**I hear you — and your feelings make complete sense.**\n\n"
                f"Watching your portfolio at {portfolio_value}{gain_str} during a "
                f"**{risk_level}** panic environment is genuinely uncomfortable. "
                f"That discomfort is real. But the feeling is telling you *to act*, "
                f"not necessarily telling you *what action to take*.\n\n"
                f"{science_line}"
                f"📊 **What actually calms me:** The panic score is **{panic_score}/100**. "
                f"During the **{similar_crash}** — a comparable moment of fear — markets recovered "
                f"in just **{recovery_months} months**. Then went up **+{post_gain}%** more.\n\n"
                f"💸 **The stakes:** Stopping your SIP of {monthly_sip} today would destroy "
                f"**{wealth_lost}** in long-term wealth — not the market, *the fear response*.\n\n"
                f"💡 **One ask:** Do nothing for 48 hours. Permanent decisions made in peak fear "
                f"almost always turn out to be mistakes. What specifically is worrying you most?"
            )

        # ══════════════════════════════════════════════════════════════════
        # ROUTE 3 — sell / exit / redeem / withdraw
        # (checked before market so "should I sell in this market?" → sell route)
        # ══════════════════════════════════════════════════════════════════
        elif any(w in msg for w in ["sell", "exit", "redeem", "withdraw", "liquidate",
                                     "pull out", "book", "square"]):
            return (
                f"{bias_preamble}"
                f"**Let me show you exactly what selling today costs — in real rupees.**\n\n"
                f"Your portfolio: **{portfolio_value}**{gain_str} (invested: {total_invested})\n\n"
                f"💸 **Two futures over {horizon_years} years:**\n"
                f"- **Hold & continue SIP** → **{hold_value}**\n"
                f"- **Sell today** → you lock in today's loss permanently, then re-enter later "
                f"at higher prices — likely reaching only **{stop_value}** or less\n"
                f"- **Wealth destroyed by exiting: {wealth_lost}**\n\n"
                f"📜 **Historical reality check:** During the **{similar_crash}**, investors who "
                f"sold at the panic bottom and re-entered 6 months later paid **40–85% more** "
                f"for the same units. The market recovered in {recovery_months} months anyway.\n\n"
                f"Selling converts a *temporary* paper loss into a *permanent* real loss — "
                f"and guarantees you miss the +{post_gain}% recovery that follows.\n\n"
                f"⏳ **My ask:** Wait 48 hours before acting. What's driving the urgency to sell today specifically?"
            )

        # ══════════════════════════════════════════════════════════════════
        # ROUTE 4 — recover / when / how long / timeline / bottom
        # (before market so "when will market recover?" hits this, not market route)
        # ══════════════════════════════════════════════════════════════════
        elif any(w in msg for w in ["recover", "recovery", "when", "how long", "timeline",
                                     "bottom", "turnaround", "rebound", "bounce"]):
            return (
                f"{bias_preamble}"
                f"**The historical record on Indian market recoveries is actually very reassuring.**\n\n"
                f"📜 **Every major Indian crash — every single one — recovered:**\n"
                f"- COVID-19 (Mar 2020): 7 months → **+112%** from bottom\n"
                f"- IL&FS / NBFC Crisis (2018): ~12 months to recovery\n"
                f"- Demonetisation shock (2016): ~3 months to recovery\n"
                f"- Global Financial Crisis (2008–09): ~18 months → **+180%**\n"
                f"- **Average recovery across all 7 crashes: ~11 months**\n\n"
                f"🎯 The current panic score is **{panic_score}/100**, most similar to "
                f"**{similar_crash}** — which recovered in **{recovery_months} months** "
                f"with **+{post_gain}%** gains from the bottom.\n\n"
                f"⚠️ **The catch:** Nobody rings a bell at the bottom. "
                f"By the time recovery feels certain, the best returns are already behind you. "
                f"The investors who captured +{post_gain}% stayed invested *during* the fear.\n\n"
                f"Your portfolio at {portfolio_value} is positioned to ride the full recovery — "
                f"but only if you stay in. Keep the {monthly_sip} SIP running."
            )

        # ══════════════════════════════════════════════════════════════════
        # ROUTE 5 — invest / buy / start / increase / opportunity
        # ══════════════════════════════════════════════════════════════════
        elif any(w in msg for w in ["invest", "buy", "start", "add", "increase", "opportunity", "lump"]):
            return (
                f"{bias_preamble}"
                f"**You're asking exactly the right question — crashes are where wealth is made.**\n\n"
                f"With a panic score of **{panic_score}/100 ({risk_level})**, we're in the kind of "
                f"environment long-term investors dream about. Units are cheap. Fear is high. "
                f"That's the classic buy signal.\n\n"
                f"📜 **Historical proof from {similar_crash}:**\n"
                f"- Recovery time: **{recovery_months} months**\n"
                f"- Gain from bottom: **+{post_gain}%**\n"
                f"- SIP continuers: built significantly more wealth than those who stopped\n\n"
                f"📊 **Your three paths over {horizon_years} years:**\n"
                f"- Stop SIP (panic): **{stop_value}**\n"
                f"- Continue SIP (hold): **{hold_value}**\n"
                f"- Increase SIP 50% (brave): **{brave_value}** ← the historically optimal move\n\n"
                f"Your current portfolio of {portfolio_value} is the base. "
                f"Every rupee you add here buys units at a discount to where they'll be in {recovery_months} months."
            )

        # ══════════════════════════════════════════════════════════════════
        # ROUTE 6 — market / crash / fall / drop (general market context)
        # ══════════════════════════════════════════════════════════════════
        elif any(w in msg for w in ["market", "crash", "fall", "drop", "nifty", "sensex", "correction", "bear"]):
            vix_context = "elevated — signalling significant fear" if vix_high else "within manageable range"
            return (
                f"{bias_preamble}"
                f"**Here's the real market picture — and some historical perspective.**\n\n"
                f"📈 Nifty 50 is at **{nifty_str}**. India VIX (fear gauge) is **{vix_str}** — {vix_context}.\n"
                f"🔴 Panic score: **{panic_score}/100 ({risk_level})**.\n\n"
                f"This environment is most similar to the **{similar_crash}**. "
                f"That crash felt catastrophic in the moment too.\n\n"
                f"📜 **What happened next:** Markets recovered in **{recovery_months} months** "
                f"and then delivered **+{post_gain}%** gains from the bottom. "
                f"India has had 7 major crashes in 20 years. All 7 recovered. Zero exceptions.\n\n"
                f"Your portfolio is at {portfolio_value}{gain_str}. The paper loss is real — "
                f"but so is the eventual recovery. The question is whether you'll be invested when it arrives.\n\n"
                f"Continuing your SIP during this period puts you in the 24% who win."
            )

        # ══════════════════════════════════════════════════════════════════
        # DEFAULT — summary of key data + HOLD recommendation
        # ══════════════════════════════════════════════════════════════════
        else:
            return (
                f"{bias_preamble}"
                f"**Here's your full picture — and my recommendation.**\n\n"
                f"📊 **Market right now:**\n"
                f"- Panic score: **{panic_score}/100 ({risk_level})**\n"
                f"- India VIX: {vix_str} | Nifty: {nifty_str}\n"
                f"- Most similar crash: **{similar_crash}** (recovered in {recovery_months} months)\n\n"
                f"💼 **Your portfolio:**\n"
                f"- Invested: {total_invested} → Current: {portfolio_value}{gain_str}\n"
                f"- Monthly SIP: {monthly_sip}\n\n"
                f"📈 **Your {horizon_years}-year outcomes:**\n"
                f"- Stop SIP (panic): {stop_value}\n"
                f"- Continue SIP (hold): **{hold_value}** ← recommended\n"
                f"- Increase SIP 50% (brave): **{brave_value}**\n"
                f"- **Cost of stopping: {wealth_lost}**\n\n"
                f"🎯 **Recommendation: HOLD and keep the SIP running.**\n"
                f"The {similar_crash} scenario recovered in {recovery_months} months and then "
                f"delivered +{post_gain}% from the bottom. Patience wins here.\n\n"
                f"What specific concern can I help you think through?"
            )

    def _fallback_message(self) -> str:
        """Absolute last-resort response."""
        return (
            "I understand this market feels scary right now. But consider this: "
            "every major crash in Indian market history has been followed by a full recovery. "
            "The investors who stayed invested came out significantly ahead. "
            "Before making any changes to your portfolio, take 48 hours and revisit this decision. "
            "What specifically is worrying you most right now?"
        )
