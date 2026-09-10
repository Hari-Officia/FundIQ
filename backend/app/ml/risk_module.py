import numpy as np
from typing import Dict, Any

class RiskModule:
    """
    Historical Risk Evaluation Module.
    Calculates Low Risk, Moderate Risk, or High Risk based on:
    - 20-day volatility (vol_20d)
    - 60-day volatility (vol_60d)
    - Downside volatility (downside_vol_20d)
    - 60-day drawdown (drawdown_60d)
    """

    @staticmethod
    def calculate_fund_risk(
        vol_20d: float,
        vol_60d: float,
        downside_vol_20d: float,
        drawdown_60d: float
    ) -> Dict[str, Any]:
        """
        Computes normalized risk metrics and risk classification.
        - Higher volatility/drawdown = higher numerical risk score (0 - 100).
        - Risk level: Low Risk (< 35), Moderate Risk (35 - 65), High Risk (> 65).
        """
        # Ensure non-negative inputs
        v20 = abs(vol_20d)
        v60 = abs(vol_60d)
        d_vol = abs(downside_vol_20d)
        dd60 = abs(drawdown_60d)  # Drawdown is typically negative, so use absolute magnitude

        # Normalized component risk scores (standard financial threshold scaling)
        # Volatility > 2.5% daily (or ~40% annualized) is very high risk
        score_v20 = min(100.0, (v20 / 0.025) * 100.0)
        score_v60 = min(100.0, (v60 / 0.025) * 100.0)
        score_dvol = min(100.0, (d_vol / 0.020) * 100.0)
        score_dd60 = min(100.0, (dd60 / 0.15) * 100.0)  # Drawdown > 15% is high risk

        # Weighted Composite Risk Score (0 - 100)
        composite_risk_score = (
            0.30 * score_v20 +
            0.25 * score_v60 +
            0.25 * score_dvol +
            0.20 * score_dd60
        )
        composite_risk_score = float(np.clip(composite_risk_score, 0.0, 100.0))

        # Risk Classification
        if composite_risk_score < 35.0:
            risk_level = "Low Risk"
        elif composite_risk_score <= 65.0:
            risk_level = "Moderate Risk"
        else:
            risk_level = "High Risk"

        return {
            "composite_risk_score": round(composite_risk_score, 2),
            "risk_level": risk_level,
            "metrics": {
                "vol_20d": round(v20, 4),
                "vol_60d": round(v60, 4),
                "downside_vol_20d": round(d_vol, 4),
                "drawdown_60d": round(dd60, 4)
            }
        }

risk_module = RiskModule()
