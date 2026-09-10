from pydantic import BaseModel
from typing import Optional, List
from datetime import date, datetime

class WatchlistItemCreate(BaseModel):
    scheme_id: int

class WatchlistItemResponse(BaseModel):
    id: int
    watchlist_id: int
    scheme_id: int
    scheme_code: int
    scheme_name: str
    current_nav: Optional[float] = None
    return_1y: Optional[float] = None

class WatchlistResponse(BaseModel):
    id: int
    name: str
    created_at: datetime
    items: List[WatchlistItemResponse] = []

class HoldingCreate(BaseModel):
    scheme_id: int
    units: float
    purchase_nav: float
    purchase_date: date

class HoldingResponse(BaseModel):
    id: int
    portfolio_id: int
    scheme_id: int
    scheme_code: int
    scheme_name: str
    units: float
    purchase_nav: float
    purchase_date: date
    current_nav: Optional[float] = None
    invested_amount: float
    current_value: float
    unrealized_gain_loss: float
    gain_loss_percentage: float

class PortfolioResponse(BaseModel):
    id: int
    name: str
    created_at: datetime
    total_invested: float
    total_current_value: float
    total_unrealized_gain: float
    total_gain_percentage: float
    holdings: List[HoldingResponse] = []
