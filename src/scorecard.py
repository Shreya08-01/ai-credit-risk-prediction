import numpy as np
import pandas as pd
from typing import Dict, Any, Union, List

class CreditScorecardEngine:
    """
    Converts model predicted default probabilities into standardized credit scores (300-850)
    and maps them to credit risk tiers and automated underwriting recommendations.
    """
    
    def __init__(self, target_base_score: float = 600.0, pdo: float = 50.0):
        """
        :param target_base_score: Score corresponding to 1:1 odds (Offset).
        :param pdo: Points to Double Odds.
        """
        self.target_base_score = target_base_score
        self.pdo = pdo
        self.factor = pdo / np.log(2.0)
        self.offset = target_base_score
        
    def probability_to_score(self, prob_default: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
        """
        Calculates credit score from probability of default:
        Score = Offset + Factor * ln((1 - p) / p)
        """
        p_clipped = np.clip(prob_default, 1e-6, 1 - 1e-6)
        odds = (1.0 - p_clipped) / p_clipped
        score = self.offset + self.factor * np.log(odds)
        score_clipped = np.clip(np.round(score), 300, 850).astype(int)
        
        if isinstance(prob_default, (float, np.float64, np.float32)):
            return int(score_clipped)
        return score_clipped

    def get_risk_tier_and_decision(self, credit_score: int, prob_default: float) -> Dict[str, Any]:
        """Maps credit score and default probability to risk tier and decision recommendation."""
        if credit_score >= 720:
            tier = "Low Risk"
            recommendation = "APPROVE"
            explanation = "Applicant demonstrates strong financial profile and low probability of default."
            color = "#28a745"
            interest_tier = "Standard Prime Rate (5.5% - 7.5%)"
        elif credit_score >= 620:
            tier = "Medium Risk"
            recommendation = "MANUAL REVIEW"
            explanation = "Moderate default risk detected. Underwriter review recommended or higher interest rate tier."
            color = "#ffc107"
            interest_tier = "Subprime / Risk-Adjusted Rate (9.5% - 14.0%)"
        else:
            tier = "High Risk"
            recommendation = "DECLINE"
            explanation = "High probability of default. Credit application exceeds acceptable risk thresholds."
            color = "#dc3545"
            interest_tier = "Not Applicable (Decline)"
            
        return {
            'credit_score': credit_score,
            'default_probability': round(float(prob_default), 4),
            'default_probability_percent': f"{prob_default * 100:.2f}%",
            'risk_tier': tier,
            'recommendation': recommendation,
            'explanation': explanation,
            'badge_color': color,
            'suggested_pricing_tier': interest_tier
        }

    def evaluate_applicant(self, prob_default: float) -> Dict[str, Any]:
        """Convenience method to score a single applicant default probability."""
        score = self.probability_to_score(prob_default)
        return self.get_risk_tier_and_decision(score, prob_default)
