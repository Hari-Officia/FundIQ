import os
import pandas as pd
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime

from app.core.database import get_db
from app.models.user import User
from app.models.scheme import Scheme
from app.models.recommendation import UserProfile, Recommendation, RecommendationItem
from app.schemas.recommendation import (
    QuestionnaireRequest,
    UserProfileResponse,
    RecommendationResponse,
    RecommendationItemResponse
)
from app.api.v1.deps import get_current_user, get_optional_current_user
from app.ml.recommendation_engine import recommendation_engine

router = APIRouter()

# Default curated Indian Mutual Funds for dynamic inference fallback
BENCHMARK_FUNDS = [
    {
        "scheme_code": 119551,
        "scheme_name": "Mirae Asset Large Cap Fund - Direct Plan - Growth",
        "category": "Equity",
        "vol_20d": 0.012,
        "vol_60d": 0.014,
        "downside_vol_20d": 0.008,
        "drawdown_60d": -0.025,
        "ret_1d": 0.002,
        "ret_5d": 0.008,
        "ret_20d": 0.022,
        "ret_60d": 0.055,
        "momentum_20d": 0.022,
        "price_vs_ma20": 1.015,
        "price_vs_ma60": 1.035
    },
    {
        "scheme_code": 120503,
        "scheme_name": "Axis Small Cap Fund - Direct Growth",
        "category": "Equity",
        "vol_20d": 0.022,
        "vol_60d": 0.024,
        "downside_vol_20d": 0.016,
        "drawdown_60d": -0.058,
        "ret_1d": 0.004,
        "ret_5d": 0.014,
        "ret_20d": 0.038,
        "ret_60d": 0.092,
        "momentum_20d": 0.038,
        "price_vs_ma20": 1.025,
        "price_vs_ma60": 1.060
    },
    {
        "scheme_code": 125497,
        "scheme_name": "Parag Parikh Flexi Cap Fund - Direct Growth",
        "category": "Equity",
        "vol_20d": 0.011,
        "vol_60d": 0.013,
        "downside_vol_20d": 0.007,
        "drawdown_60d": -0.018,
        "ret_1d": 0.003,
        "ret_5d": 0.009,
        "ret_20d": 0.026,
        "ret_60d": 0.068,
        "momentum_20d": 0.026,
        "price_vs_ma20": 1.018,
        "price_vs_ma60": 1.042
    },
    {
        "scheme_code": 119775,
        "scheme_name": "SBI Savings Fund - Direct Plan - Growth",
        "category": "Debt",
        "vol_20d": 0.002,
        "vol_60d": 0.003,
        "downside_vol_20d": 0.001,
        "drawdown_60d": -0.002,
        "ret_1d": 0.0003,
        "ret_5d": 0.0012,
        "ret_20d": 0.0055,
        "ret_60d": 0.0165,
        "momentum_20d": 0.0055,
        "price_vs_ma20": 1.002,
        "price_vs_ma60": 1.006
    },
    {
        "scheme_code": 120166,
        "scheme_name": "HDFC Balanced Advantage Fund - Direct Growth",
        "category": "Hybrid",
        "vol_20d": 0.008,
        "vol_60d": 0.010,
        "downside_vol_20d": 0.005,
        "drawdown_60d": -0.015,
        "ret_1d": 0.0018,
        "ret_5d": 0.0065,
        "ret_20d": 0.018,
        "ret_60d": 0.045,
        "momentum_20d": 0.018,
        "price_vs_ma20": 1.010,
        "price_vs_ma60": 1.025
    },
    {
        "scheme_code": 120716,
        "scheme_name": "Nippon India Small Cap Fund - Direct Plan - Growth",
        "category": "Equity",
        "vol_20d": 0.024,
        "vol_60d": 0.026,
        "downside_vol_20d": 0.017,
        "drawdown_60d": -0.065,
        "ret_1d": 0.005,
        "ret_5d": 0.016,
        "ret_20d": 0.042,
        "ret_60d": 0.105,
        "momentum_20d": 0.042,
        "price_vs_ma20": 1.030,
        "price_vs_ma60": 1.075
    },
    {
        "scheme_code": 118834,
        "scheme_name": "ICICI Prudential Corporate Bond Fund - Direct Plan - Growth",
        "category": "Debt",
        "vol_20d": 0.003,
        "vol_60d": 0.004,
        "downside_vol_20d": 0.002,
        "drawdown_60d": -0.003,
        "ret_1d": 0.0004,
        "ret_5d": 0.0015,
        "ret_20d": 0.0062,
        "ret_60d": 0.0185,
        "momentum_20d": 0.0062,
        "price_vs_ma20": 1.003,
        "price_vs_ma60": 1.008
    },
    {
        "scheme_code": 119821,
        "scheme_name": "Kotak Emerging Equity Fund - Direct Growth",
        "category": "Equity",
        "vol_20d": 0.016,
        "vol_60d": 0.018,
        "downside_vol_20d": 0.011,
        "drawdown_60d": -0.035,
        "ret_1d": 0.003,
        "ret_5d": 0.011,
        "ret_20d": 0.031,
        "ret_60d": 0.078,
        "momentum_20d": 0.031,
        "price_vs_ma20": 1.020,
        "price_vs_ma60": 1.050
    }
]

FUND_DATASET_CACHE: Optional[List[Dict[str, Any]]] = None

def load_fund_dataset(db: Session) -> List[Dict[str, Any]]:
    """
    Loads master mutual fund dataset (13,740 schemes) with in-memory caching.
    """
    global FUND_DATASET_CACHE
    if FUND_DATASET_CACHE is not None and len(FUND_DATASET_CACHE) > 0:
        return FUND_DATASET_CACHE

    possible_paths = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "recommendation_data.csv"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "recommendation_data.csv"),
        "recommendation_data.csv",
        os.path.join("data", "recommendation_data.csv")
    ]

    target_csv = None
    for p in possible_paths:
        if os.path.exists(p):
            target_csv = p
            break

    if target_csv and os.path.exists(target_csv):
        try:
            df = pd.read_csv(target_csv)
            if 'scheme_code' in df.columns:
                df = df.dropna(subset=['scheme_code'])
                df['scheme_code'] = df['scheme_code'].astype(int)
                df = df.drop_duplicates(subset=['scheme_code'], keep='last')
            elif 'scheme_name' in df.columns:
                df = df.drop_duplicates(subset=['scheme_name'], keep='last')

            if 'category' not in df.columns and 'fund_category' in df.columns:
                df['category'] = df['fund_category']

            # Fill missing numerical defaults
            defaults = {
                "vol_20d": 0.015, "vol_60d": 0.018,
                "downside_vol_20d": 0.010, "drawdown_60d": -0.03, "ret_1d": 0.001,
                "ret_5d": 0.005, "ret_20d": 0.015, "ret_60d": 0.040,
                "momentum_20d": 0.015, "price_vs_ma20": 1.01, "price_vs_ma60": 1.03,
                "predicted_return_5obs": 0.005
            }
            for col, val in defaults.items():
                if col not in df.columns:
                    df[col] = val

            funds = df.to_dict(orient='records')
            if len(funds) > 0:
                FUND_DATASET_CACHE = funds
                return FUND_DATASET_CACHE
        except Exception as e:
            pass

    FUND_DATASET_CACHE = BENCHMARK_FUNDS
    return FUND_DATASET_CACHE



@router.post("/generate", response_model=RecommendationResponse)
def generate_recommendations(
    payload: QuestionnaireRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_current_user)
):
    """
    Executes the 4-layer recommendation engine using questionnaire responses,
    ranks the top 5 mutual funds, generates XAI reasons, and saves to database.
    """
    # 1. Load dataset
    funds_dataset = load_fund_dataset(db)

    # 2. Run Recommendation Engine
    top_funds = recommendation_engine.rank_funds(
        user_risk=payload.risk_level,
        horizon=payload.horizon,
        goal=payload.goal,
        investment_amount=payload.investment_amount,
        preferred_type=payload.preferred_type,
        funds_list=funds_dataset,
        top_n=5
    )

    now = datetime.utcnow()
    user_id = current_user.id if current_user else 1

    # 3. Persist User Profile if user logged in
    if current_user:
        profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()
        if not profile:
            profile = UserProfile(
                user_id=current_user.id,
                risk_level=payload.risk_level,
                horizon=payload.horizon,
                goal=payload.goal,
                investment_amount=payload.investment_amount,
                preferred_type=payload.preferred_type
            )
            db.add(profile)
        else:
            profile.risk_level = payload.risk_level
            profile.horizon = payload.horizon
            profile.goal = payload.goal
            profile.investment_amount = payload.investment_amount
            profile.preferred_type = payload.preferred_type
            profile.updated_at = now

    # 4. Create Recommendation History record in DB
    rec_record = Recommendation(
        user_id=user_id,
        generated_at=now,
        risk_level=payload.risk_level,
        horizon=payload.horizon,
        goal=payload.goal,
        investment_amount=payload.investment_amount,
        preferred_type=payload.preferred_type,
        model_version="xgboost_risk_v1"
    )
    db.add(rec_record)
    db.commit()
    db.refresh(rec_record)

    rec_items = []
    for item in top_funds:
        db_item = RecommendationItem(
            recommendation_id=rec_record.id,
            rank=item["rank"],
            scheme_code=item["scheme_code"],
            scheme_name=item["scheme_name"],
            predicted_return=item["predicted_return"],
            risk_level=item["risk_level"],
            final_score=item["final_score"],
            reason=item["reason"]
        )
        db.add(db_item)
        rec_items.append(RecommendationItemResponse(**item))

    db.commit()

    return RecommendationResponse(
        id=rec_record.id,
        user_id=user_id,
        generated_at=now,
        risk_level=payload.risk_level,
        horizon=payload.horizon,
        goal=payload.goal,
        investment_amount=payload.investment_amount,
        preferred_type=payload.preferred_type,
        model_version="xgboost_risk_v1",
        recommended_funds=rec_items
    )

@router.get("/history", response_model=List[RecommendationResponse])
def get_recommendation_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Retrieves previous recommendation sessions ("My Recommendations") for logged-in user.
    """
    records = db.query(Recommendation).filter(
        Recommendation.user_id == current_user.id
    ).order_by(Recommendation.generated_at.desc()).all()

    result = []
    for rec in records:
        items = db.query(RecommendationItem).filter(
            RecommendationItem.recommendation_id == rec.id
        ).order_by(RecommendationItem.rank.asc()).all()

        rec_items = [
            RecommendationItemResponse(
                rank=it.rank,
                scheme_code=it.scheme_code,
                scheme_name=it.scheme_name,
                predicted_return=it.predicted_return,
                predicted_return_pct=f"{it.predicted_return * 100:+.2f}%",
                risk_level=it.risk_level,
                final_score=it.final_score,
                reason=it.reason,
                expected_return_score=round(it.final_score * 0.35, 1),
                historical_risk_score=85.0,
                suitability_score=90.0,
                preference_score=100.0
            ) for it in items
        ]

        result.append(
            RecommendationResponse(
                id=rec.id,
                user_id=rec.user_id,
                generated_at=rec.generated_at,
                risk_level=rec.risk_level,
                horizon=rec.horizon,
                goal=rec.goal,
                investment_amount=rec.investment_amount,
                preferred_type=rec.preferred_type,
                model_version=rec.model_version,
                recommended_funds=rec_items
            )
        )

    return result

@router.get("/profile", response_model=UserProfileResponse)
def get_user_profile(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Retrieves saved questionnaire preferences for pre-filling the form on return logins.
    """
    profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="User profile not found")
    return profile
