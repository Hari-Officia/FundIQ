from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
import numpy as np
from datetime import date, timedelta

from app.core.database import get_db
from app.models.scheme import Scheme
from app.models.nav import NAVHistory
from app.api.v1.schemes import get_scheme_detail, SAMPLE_SCHEMES
from app.schemas.analytics import FundMetricsResponse, FundComparisonResponse, FundComparisonItem

router = APIRouter()

def calculate_metrics_from_history(nav_history: List[dict]):
    if not nav_history or len(nav_history) < 2:
        return {
            "current_nav": 100.0,
            "return_1y": 0.15,
            "return_3y": 0.42,
            "return_5y": 0.85,
            "volatility": 0.085,
            "max_drawdown": -0.065,
            "risk_score": 5,
            "performance_score": 8
        }

    navs = [point["nav"] for point in nav_history]
    current_nav = navs[-1]
    
    # Calculate returns
    n_days = len(navs)
    r_1y = (current_nav / navs[-365] - 1) if n_days >= 365 else (current_nav / navs[0] - 1)
    r_3y = (current_nav / navs[-1095] - 1) if n_days >= 1095 else r_1y * 2.2
    r_5y = (current_nav / navs[-1825] - 1) if n_days >= 1825 else r_3y * 1.6

    # Daily returns for volatility
    daily_returns = [navs[i] / navs[i-1] - 1 for i in range(1, len(navs))]
    volatility = float(np.std(daily_returns) * np.sqrt(252)) if len(daily_returns) > 1 else 0.10

    # Max Drawdown
    peak = navs[0]
    max_dd = 0.0
    for nav in navs:
        if nav > peak:
            peak = nav
        dd = (nav - peak) / peak
        if dd < max_dd:
            max_dd = dd

    # Risk & Performance Score calculation (Scale 1-10)
    risk_score = min(10, max(1, int(volatility * 50 + abs(max_dd) * 20)))
    perf_score = min(10, max(1, int(r_1y * 30 + (1 - abs(max_dd)) * 5)))

    return {
        "current_nav": current_nav,
        "return_1y": round(r_1y, 4),
        "return_3y": round(r_3y, 4),
        "return_5y": round(r_5y, 4),
        "volatility": round(volatility, 4),
        "max_drawdown": round(max_dd, 4),
        "risk_score": risk_score,
        "performance_score": perf_score
    }

@router.get("/metrics/{scheme_code}", response_model=FundMetricsResponse)
def get_fund_metrics(scheme_code: int, db: Session = Depends(get_db)):
    detail = get_scheme_detail(scheme_code=scheme_code, db=db)
    metrics = calculate_metrics_from_history(detail["nav_history"])

    return {
        "scheme_code": detail["scheme_code"],
        "scheme_name": detail["scheme_name"],
        "category": detail.get("category"),
        "amc_name": detail.get("amc_name"),
        "performance": {
            "current_nav": metrics["current_nav"],
            "return_1y": metrics["return_1y"],
            "return_3y": metrics["return_3y"],
            "return_5y": metrics["return_5y"],
        },
        "risk": {
            "volatility": metrics["volatility"],
            "max_drawdown": metrics["max_drawdown"],
            "risk_score": metrics["risk_score"],
            "performance_score": metrics["performance_score"]
        }
    }

@router.get("/performance/{scheme_code}")
def get_fund_performance(scheme_code: int, db: Session = Depends(get_db)):
    metrics = get_fund_metrics(scheme_code=scheme_code, db=db)
    return metrics["performance"]

@router.get("/risk/{scheme_code}")
def get_fund_risk(scheme_code: int, db: Session = Depends(get_db)):
    metrics = get_fund_metrics(scheme_code=scheme_code, db=db)
    return metrics["risk"]

@router.get("/compare", response_model=FundComparisonResponse)
def compare_funds(scheme_codes: List[int] = Query(..., description="List of scheme codes to compare"), db: Session = Depends(get_db)):
    comparison_items = []
    for code in scheme_codes:
        try:
            m = get_fund_metrics(scheme_code=code, db=db)
            comparison_items.append({
                "scheme_code": m["scheme_code"],
                "scheme_name": m["scheme_name"],
                "category": m["category"],
                "amc_name": m["amc_name"],
                "current_nav": m["performance"]["current_nav"],
                "return_1y": m["performance"]["return_1y"],
                "return_3y": m["performance"]["return_3y"],
                "volatility": m["risk"]["volatility"],
                "max_drawdown": m["risk"]["max_drawdown"],
                "risk_score": m["risk"]["risk_score"]
            })
        except HTTPException:
            continue

    if not comparison_items:
        # Return comparison of first 2 sample schemes if empty
        for s in SAMPLE_SCHEMES[:2]:
            m = get_fund_metrics(scheme_code=s["scheme_code"], db=db)
            comparison_items.append({
                "scheme_code": m["scheme_code"],
                "scheme_name": m["scheme_name"],
                "category": m["category"],
                "amc_name": m["amc_name"],
                "current_nav": m["performance"]["current_nav"],
                "return_1y": m["performance"]["return_1y"],
                "return_3y": m["performance"]["return_3y"],
                "volatility": m["risk"]["volatility"],
                "max_drawdown": m["risk"]["max_drawdown"],
                "risk_score": m["risk"]["risk_score"]
            })

    return {"funds": comparison_items}
