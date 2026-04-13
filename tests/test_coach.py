"""
Tests: Behavioral Coach Bias Detection
"""

import pytest
from agents.behavioral_coach import BehavioralCoach


class TestBiasDetection:
    def setup_method(self):
        self.coach = BehavioralCoach()

    def test_herd_behavior_detection(self):
        msg = "Everyone is selling their funds right now"
        bias = self.coach._detect_bias(msg)
        assert bias == "Herd Behavior"

    def test_loss_aversion_detection(self):
        msg = "I'm losing so much money, I can't take it anymore"
        bias = self.coach._detect_bias(msg)
        assert bias == "Loss Aversion"

    def test_recency_bias_detection(self):
        msg = "The market always falls in April, it never recovers"
        bias = self.coach._detect_bias(msg)
        assert bias == "Recency Bias"

    def test_anchoring_detection(self):
        msg = "My Nifty ETF bought at 22,000 and was at 24,500 just last month"
        bias = self.coach._detect_bias(msg)
        assert bias == "Anchoring"

    def test_availability_heuristic(self):
        msg = "I saw on news that the market is going to crash further"
        bias = self.coach._detect_bias(msg)
        assert bias == "Availability Heuristic"

    def test_no_bias_normal_message(self):
        msg = "What should I know about SIP continuity?"
        bias = self.coach._detect_bias(msg)
        assert bias == ""

    def test_reset_clears_history(self):
        self.coach.conversation_history = [{"role": "user", "content": "test"}]
        self.coach.reset()
        assert len(self.coach.conversation_history) == 0
