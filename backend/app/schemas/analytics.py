from pydantic import BaseModel
from typing import Optional, List, Dict

class PerformanceMetrics(BaseModel):
    current_nav: float
    return_1y: Optional[float] = None
    return_3y: Optional[float] = None
    return_5y: Optional[float] = None

class RiskMetrics(BaseModel):
    volatility: Optional[float] = None
    max_drawdown: Optional[float] = None
    risk_score: Optional[int] = None      # Scale 1-10
    performance_score: Optional[int] = None # Scale 1-10

class FundMetricsResponse(BaseModel):
    scheme_code: int
    scheme_name: str
    category: Optional[str] = None
    amc_name: Optional[str] = None
    performance: PerformanceMetrics
    risk: RiskMetrics

class FundComparisonItem(BaseModel):
    scheme_code: int
    scheme_name: str
    category: Optional[str] = None
    amc_name: Optional[str] = None
    current_nav: float
    return_1y: Optional[float] = None
    return_3y: Optional[float] = None
    volatility: Optional[float] = None
    max_drawdown: Optional[float] = None
    risk_score: Optional[int] = None

class FundComparisonResponse(BaseModel):
    funds: List[FundComparisonItem]
