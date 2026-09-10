from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class QuestionnaireRequest(BaseModel):
    risk_level: str = Field(..., description="Low, Moderate, High")
    horizon: str = Field(..., description="< 1 year, 1-3 years, 3-5 years, > 5 years")
    goal: str = Field(..., description="Capital preservation, Regular income, Wealth growth, Tax saving")
    investment_amount: float = Field(..., gt=0, description="Investment amount in INR")
    preferred_type: str = Field(..., description="Equity, Debt, Hybrid, No preference")
    investment_style: Optional[str] = Field(None, description="Higher growth potential, Lower risk, Balanced")

class UserProfileResponse(BaseModel):
    user_id: int
    risk_level: str
    horizon: str
    goal: str
    investment_amount: float
    preferred_type: str
    updated_at: datetime

    class Config:
        from_attributes = True

class RecommendationItemResponse(BaseModel):
    rank: int
    scheme_code: int
    scheme_name: str
    detected_type: Optional[str] = "Other"
    category: Optional[str] = "Other"
    fund_risk: Optional[str] = "Moderate"
    risk_level: str
    predicted_return: float
    predicted_return_pct: Optional[str] = "0.000%"
    final_score: float
    match_score_pct: Optional[str] = "0.0%"
    reason: str
    expected_return_score: Optional[float] = 0.0
    historical_risk_score: Optional[float] = 0.0
    suitability_score: Optional[float] = 0.0
    preference_score: Optional[float] = 0.0

    class Config:
        from_attributes = True


class RecommendationResponse(BaseModel):
    id: Optional[int] = None
    user_id: int
    generated_at: datetime
    risk_level: str
    horizon: str
    goal: str
    investment_amount: float
    preferred_type: str
    model_version: str
    recommended_funds: List[RecommendationItemResponse]

    model_config = {
        "from_attributes": True,
        "protected_namespaces": ()
    }

