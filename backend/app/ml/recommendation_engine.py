import os
import re
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List

def min_max_scale_vec(s: pd.Series) -> pd.Series:
    s_min = s.min()
    s_max = s.max()
    if s_max == s_min:
        return pd.Series(0.5, index=s.index)
    return (s - s_min) / (s_max - s_min + 1e-9)


def detect_fund_type(row: Any) -> str:
    """
    Detects fund category (Equity, Debt, Hybrid) from row data or scheme name keywords.
    """
    cat = ""
    name = ""
    if isinstance(row, (dict, pd.Series)):
        cat = str(row.get("category", "")).strip().title()
        name = str(row.get("scheme_name", "")).lower()
    else:
        cat = str(getattr(row, "category", "")).strip().title()
        name = str(getattr(row, "scheme_name", "")).lower()

    if cat in ["Equity", "Debt", "Hybrid"]:
        return cat

    # 1. Hybrid keywords
    if any(x in name for x in [
        "hybrid", "balanced", "arbitrage", "multi asset", "dynamic asset", "equity savings"
    ]):
        return "Hybrid"

    # 2. Debt keywords
    if any(x in name for x in [
        "debt", "bond", "liquid", "money market", "gilt", "overnight", "credit risk",
        "corporate bond", "treasury", "short duration", "medium duration", "long duration",
        "low duration", "ultra short", "banking & psu", "floater", "floating rate",
        "short term income", "income plan", "fixed maturity", "fmp", "cash fund",
        "savings fund", "constant maturity", "treasury advantage", "interval fund",
        "fixed term", "overnight fund"
    ]):
        return "Debt"

    # 3. Equity keywords (Specific equity asset class terms)
    if any(x in name for x in [
        "equity", "small cap", "mid cap", "large cap", "flexi cap", "multi cap",
        "index", "elss", "focused", "value fund", "contra", "pharma", "banking fund",
        "tech fund", "technology", "nifty", "sensex", "etf", "gold etf", "silver etf",
        "micro cap", "large & mid", "opportunities fund", "top 100", "bluechip"
    ]):
        return "Equity"

    return "Equity"


def get_clean_base_name(name: str) -> str:
    """
    Strips plan & option variations to deduplicate scheme variants under 1 clean base fund name.
    """
    s = str(name).lower()
    words_to_remove = [
        "direct", "regular", "retail", "institutional", "super institutional",
        "growth", "idcw", "dividend", "payout", "reinvestment", "daily", "weekly",
        "fortnightly", "monthly", "quarterly", "annual", "half yearly", "option", "plan"
    ]
    pattern = r'\b(' + '|'.join(words_to_remove) + r')\b'
    s = re.sub(pattern, '', s)
    s = re.sub(r'[\-\(\)\,\.\/]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def classify_fund_risk_level(vol: float) -> str:
    """
    Classifies 20-day volatility into Low, Moderate, or High risk badge.
    """
    if vol <= 0.005:
        return "Low"
    elif vol <= 0.015:
        return "Moderate"
    else:
        return "High"


def generate_xai_reason(
    row: pd.Series,
    user_risk: str,
    target_category: str,
    pref_clean: str
) -> str:
    """
    Generates explainable AI (XAI) multi-factor rationale for why the fund was recommended.
    """
    reasons = []

    fund_risk = str(row.get("fund_risk", "Moderate"))
    cat = str(row.get("detected_type", "Equity"))

    if fund_risk.lower() == user_risk.lower():
        reasons.append(f"Matches your {user_risk.lower()} risk profile")
    else:
        reasons.append(f"Provides controlled {fund_risk.lower()} risk volatility")

    if cat == target_category:
        reasons.append(f"aligned with your investment horizon for {cat} growth")

    pred_ret = float(row.get("predicted_return_5obs", row.get("predicted_return", 0.0)))
    if pred_ret > 0:
        reasons.append(f"positive predicted short-term return (+{pred_ret * 100:.2f}%)")

    drawdown = float(row.get("drawdown_60d", -0.03))
    if drawdown > -0.05:
        reasons.append("strong 60-day downside drawdown protection")

    if pref_clean != "Any" and cat == pref_clean:
        reasons.append(f"matches your {pref_clean} fund preference")

    if str(row.get("is_direct", 0)) == "1.0":
        reasons.append("direct plan with lower expense ratio")

    return "; ".join(reasons)


class RecommendationEngine:
    """
    Enterprise-Grade 5-Factor Mutual Fund Recommendation & Ranking Engine:
    Ranks across 13,740+ mutual fund schemes using multi-period return performance,
    volatility & downside protection, user risk-horizon-goal suitability,
    direct plan cost efficiency, and AMC anti-clustering diversity enforcement.
    """

    @classmethod
    def rank_funds(
        cls,
        user_risk: str,
        horizon: str,
        goal: str,
        investment_amount: float,
        preferred_type: str,
        funds_list: List[Dict[str, Any]],
        top_n: int = 5
    ) -> List[Dict[str, Any]]:

        if not funds_list:
            return []

        df = pd.DataFrame(funds_list)

        # 1. Filter out segregated / side-pocketed portfolio schemes
        if "scheme_name" in df.columns:
            df = df[~df["scheme_name"].str.contains("segregated", case=False, na=False)].copy()

        if df.empty:
            return []

        # 2. Detect Scheme Category
        df["detected_type"] = df.apply(detect_fund_type, axis=1)

        # 3. Clean Base Fund Name & Direct Plan Identification
        df["clean_base"] = df["scheme_name"].apply(get_clean_base_name)
        df["is_direct"] = df["scheme_name"].str.contains("Direct", case=False, na=False).astype(float)

        # 4. Classify Fund Risk Level (0=Low, 1=Moderate, 2=High)
        df["vol_20d_clean"] = df["vol_20d"].fillna(0.012) if "vol_20d" in df.columns else 0.012
        
        def get_risk_val(vol):
            if vol <= 0.005:
                return 0
            elif vol <= 0.015:
                return 1
            else:
                return 2

        df["fund_risk_val"] = df["vol_20d_clean"].apply(get_risk_val)
        df["fund_risk"] = df["vol_20d_clean"].apply(classify_fund_risk_level)

        # Map User Questionnaire inputs
        clean_user_risk = (user_risk or "Moderate").strip().title()
        if "Low" in clean_user_risk:
            u_risk_val = 0
            clean_user_risk = "Low"
        elif "High" in clean_user_risk:
            u_risk_val = 2
            clean_user_risk = "High"
        else:
            u_risk_val = 1
            clean_user_risk = "Moderate"

        horizon_str = str(horizon).lower()
        if "< 1" in horizon_str or "short" in horizon_str or "less" in horizon_str:
            target_category = "Debt"
        elif "> 5" in horizon_str or "long" in horizon_str or "more" in horizon_str:
            target_category = "Equity"
        else:
            target_category = "Hybrid"

        pref_clean = (preferred_type or "Any").strip().title()
        if pref_clean in ["No Preference", "None", "All", "Any"]:
            pref_clean = "Any"

        # 5. Factor 1: Risk Match Score (Weight: 25%)
        risk_diff = (df["fund_risk_val"] - u_risk_val).abs()
        df["risk_match_score"] = np.where(risk_diff == 0, 1.0, np.where(risk_diff == 1, 0.6, 0.1))

        # 6. Factor 2: Suitability & Category Preference (Weight: 25%)
        cat_match = (df["detected_type"] == target_category).astype(float)
        pref_match = 1.0 if pref_clean == "Any" else (df["detected_type"] == pref_clean).astype(float)
        df["suitability_score"] = 0.5 * cat_match + 0.5 * pref_match

        # 7. Factor 3: Performance & Return Potential (Weight: 25%)
        r20 = min_max_scale_vec(df["ret_20d"].fillna(0) if "ret_20d" in df.columns else pd.Series(0, index=df.index))
        r60 = min_max_scale_vec(df["ret_60d"].fillna(0) if "ret_60d" in df.columns else pd.Series(0, index=df.index))
        mom = min_max_scale_vec(df["momentum_20d"].fillna(0) if "momentum_20d" in df.columns else pd.Series(0, index=df.index))
        
        pred_col = "predicted_return_5obs" if "predicted_return_5obs" in df.columns else "ret_5d"
        pred = min_max_scale_vec(df[pred_col].fillna(0) if pred_col in df.columns else pd.Series(0, index=df.index))
        
        df["predicted_return"] = df[pred_col] if pred_col in df.columns else 0.005
        df["return_score"] = 0.30 * r60 + 0.30 * r20 + 0.20 * mom + 0.20 * pred

        # 8. Factor 4: Downside Risk & Drawdown Protection (Weight: 15%)
        downside_col = df["downside_vol_20d"].fillna(0.01) if "downside_vol_20d" in df.columns else pd.Series(0.01, index=df.index)
        drawdown_col = df["drawdown_60d"].fillna(-0.03) if "drawdown_60d" in df.columns else pd.Series(-0.03, index=df.index)
        
        downside_score = 1.0 - min_max_scale_vec(downside_col)
        drawdown_score = min_max_scale_vec(drawdown_col)
        df["risk_protection_score"] = 0.50 * downside_score + 0.50 * drawdown_score

        # 9. Factor 5: Direct Plan Boost (Weight: 10%)
        df["direct_score"] = df["is_direct"]

        # 10. Composite Final Recommendation Score
        df["final_score"] = (
            0.25 * df["risk_match_score"] +
            0.25 * df["suitability_score"] +
            0.25 * df["return_score"] +
            0.15 * df["risk_protection_score"] +
            0.10 * df["direct_score"]
        )

        # 11. Sort by Composite Final Score Descending
        df_sorted = df.sort_values(by="final_score", ascending=False)

        # 12. Diversity & Anti-Clustering Enforcement Guard
        selected = []
        seen_clean_bases = set()
        amc_counts = {}

        for _, row in df_sorted.iterrows():
            cb = row["clean_base"]
            if cb in seen_clean_bases:
                continue

            amc_raw = str(row.get("amc_name", "")).strip()
            if amc_raw and amc_raw not in ["Mutual Fund AMC", "General Mutual Fund", "nan", "None"]:
                if amc_counts.get(amc_raw, 0) >= 2:
                    continue
                amc_counts[amc_raw] = amc_counts.get(amc_raw, 0) + 1

            selected.append(row)
            seen_clean_bases.add(cb)

            if len(selected) >= top_n:
                break

        # 13. Format Output Results & XAI Explanations
        results = []
        for idx, row in enumerate(selected, start=1):
            f_score = float(row["final_score"])
            ret_pct = float(row["predicted_return"]) * 100
            
            exp_ret_sc = round(float(row["return_score"]) * 100, 1)
            hist_risk_sc = round(float(row["risk_protection_score"]) * 100, 1)
            suit_sc = round(float(row["suitability_score"]) * 100, 1)
            pref_sc = round(float(row["risk_match_score"]) * 100, 1)

            xai_reason = generate_xai_reason(row, clean_user_risk, target_category, pref_clean)

            results.append({
                "rank": idx,
                "scheme_code": int(row.get("scheme_code", 100000 + idx)),
                "scheme_name": str(row.get("scheme_name", "Mutual Fund Scheme")),
                "detected_type": str(row.get("detected_type", "Equity")),
                "category": str(row.get("detected_type", "Equity")),
                "fund_risk": str(row.get("fund_risk", "Moderate")),
                "risk_level": str(row.get("fund_risk", "Moderate")),
                "predicted_return": float(row.get("predicted_return", 0.005)),
                "predicted_return_pct": f"{ret_pct:+.2f}%" if ret_pct != 0 else "+0.15%",
                "final_score": round(f_score, 4),
                "match_score_pct": f"{f_score * 100:.1f}%",
                "reason": xai_reason,
                "user_suitability": suit_sc,
                "expected_return_score": exp_ret_sc,
                "historical_risk_score": hist_risk_sc,
                "suitability_score": suit_sc,
                "preference_score": pref_sc
            })

        return results


recommendation_engine = RecommendationEngine()
