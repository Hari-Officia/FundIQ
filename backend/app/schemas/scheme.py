from pydantic import BaseModel
from typing import Optional, List
from datetime import date

class NAVHistoryPoint(BaseModel):
    date: date
    nav: float

    class Config:
        from_attributes = True

class SchemeBase(BaseModel):
    scheme_code: int
    scheme_name: str
    isin: Optional[str] = None
    amc_code: Optional[str] = None
    amc_name: Optional[str] = None
    category: Optional[str] = None

class SchemeCreate(SchemeBase):
    pass

class SchemeResponse(SchemeBase):
    id: int
    current_nav: Optional[float] = None
    latest_nav_date: Optional[date] = None

    class Config:
        from_attributes = True

class SchemeDetailResponse(SchemeResponse):
    nav_history: List[NAVHistoryPoint] = []
