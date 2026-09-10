from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Dict

class PredictionRequest(BaseModel):
    scheme_code: int
    horizon_days: int = 30  # 1, 7, or 30 days
    model: Optional[str] = "ensemble"  # random_forest, xgboost, or ensemble

class FactorDriver(BaseModel):
    factor: str
    impact: str  # 'positive', 'negative', 'neutral'
    description: str

class ModelBreakdown(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    random_forest_return: float
    xgboost_return: float
    ensemble_return: float

class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    scheme_code: int
    scheme_name: str
    current_nav: float
    horizon_days: int
    predicted_return: float  # e.g., 0.0421 = +4.21%
    predicted_nav: float     # e.g., 108.74
    model: str
    model_version: str = "xgb_rf_v1"
    model_breakdown: ModelBreakdown
    factors: List[FactorDriver]
    disclaimer: str = "Model estimate for analytical purposes. Not guaranteed returns or buy/sell advice."

