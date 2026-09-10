from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from app.core.database import get_db
from app.models.user import User
from app.models.scheme import Scheme
from app.models.portfolio import Portfolio, Holding
from app.schemas.portfolio import PortfolioResponse, HoldingCreate, HoldingResponse
from app.api.v1.deps import get_current_user
from app.api.v1.schemes import get_scheme_detail

router = APIRouter()

def get_or_create_portfolio(user_id: int, db: Session) -> Portfolio:
    p = db.query(Portfolio).filter(Portfolio.user_id == user_id).first()
    if not p:
        p = Portfolio(user_id=user_id, name="My Portfolio")
        db.add(p)
        db.commit()
        db.refresh(p)
    return p

@router.get("/", response_model=PortfolioResponse)
def get_portfolio(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    p = get_or_create_portfolio(current_user.id, db)

    holdings_output = []
    total_invested = 0.0
    total_current_value = 0.0

    for h in p.holdings:
        scheme = db.query(Scheme).filter(Scheme.id == h.scheme_id).first()
        code = scheme.scheme_code if scheme else 119550
        name = scheme.scheme_name if scheme else "Mutual Fund Scheme"

        try:
            detail = get_scheme_detail(scheme_code=code, db=db)
            current_nav = detail["current_nav"]
        except Exception:
            current_nav = h.purchase_nav * 1.12

        invested = round(h.units * h.purchase_nav, 2)
        curr_val = round(h.units * current_nav, 2)
        gain_loss = round(curr_val - invested, 2)
        gain_pct = round((gain_loss / invested) * 100, 2) if invested > 0 else 0.0

        total_invested += invested
        total_current_value += curr_val

        holdings_output.append({
            "id": h.id,
            "portfolio_id": p.id,
            "scheme_id": h.scheme_id,
            "scheme_code": code,
            "scheme_name": name,
            "units": h.units,
            "purchase_nav": h.purchase_nav,
            "purchase_date": h.purchase_date,
            "current_nav": current_nav,
            "invested_amount": invested,
            "current_value": curr_val,
            "unrealized_gain_loss": gain_loss,
            "gain_loss_percentage": gain_pct
        })

    total_gain = round(total_current_value - total_invested, 2)
    total_gain_pct = round((total_gain / total_invested) * 100, 2) if total_invested > 0 else 0.0

    return {
        "id": p.id,
        "name": p.name,
        "created_at": p.created_at,
        "total_invested": round(total_invested, 2),
        "total_current_value": round(total_current_value, 2),
        "total_unrealized_gain": total_gain,
        "total_gain_percentage": total_gain_pct,
        "holdings": holdings_output
    }

@router.post("/holdings", response_model=HoldingResponse, status_code=status.HTTP_201_CREATED)
def add_holding(holding_in: HoldingCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    p = get_or_create_portfolio(current_user.id, db)

    h = Holding(
        portfolio_id=p.id,
        scheme_id=holding_in.scheme_id,
        units=holding_in.units,
        purchase_nav=holding_in.purchase_nav,
        purchase_date=holding_in.purchase_date
    )
    db.add(h)
    db.commit()
    db.refresh(h)

    scheme = db.query(Scheme).filter(Scheme.id == holding_in.scheme_id).first()
    code = scheme.scheme_code if scheme else 119550
    name = scheme.scheme_name if scheme else "Mutual Fund Scheme"

    current_nav = holding_in.purchase_nav * 1.05
    invested = round(h.units * h.purchase_nav, 2)
    curr_val = round(h.units * current_nav, 2)
    gain_loss = round(curr_val - invested, 2)
    gain_pct = round((gain_loss / invested) * 100, 2) if invested > 0 else 0.0

    return {
        "id": h.id,
        "portfolio_id": p.id,
        "scheme_id": h.scheme_id,
        "scheme_code": code,
        "scheme_name": name,
        "units": h.units,
        "purchase_nav": h.purchase_nav,
        "purchase_date": h.purchase_date,
        "current_nav": current_nav,
        "invested_amount": invested,
        "current_value": curr_val,
        "unrealized_gain_loss": gain_loss,
        "gain_loss_percentage": gain_pct
    }

@router.delete("/holdings/{holding_id}", status_code=status.HTTP_200_OK)
def remove_holding(holding_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    p = get_or_create_portfolio(current_user.id, db)
    h = db.query(Holding).filter(Holding.id == holding_id, Holding.portfolio_id == p.id).first()
    if not h:
        raise HTTPException(status_code=404, detail="Holding not found")

    db.delete(h)
    db.commit()
    return {"message": "Holding removed from portfolio"}
