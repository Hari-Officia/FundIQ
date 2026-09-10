from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from app.core.database import get_db
from app.models.user import User
from app.models.scheme import Scheme
from app.models.nav import NAVHistory
from app.models.portfolio import Watchlist, WatchlistItem
from app.schemas.portfolio import WatchlistResponse, WatchlistItemCreate, WatchlistItemResponse
from app.api.v1.deps import get_current_user
from app.api.v1.schemes import get_scheme_detail

router = APIRouter()

def get_or_create_watchlist(user_id: int, db: Session) -> Watchlist:
    w = db.query(Watchlist).filter(Watchlist.user_id == user_id).first()
    if not w:
        w = Watchlist(user_id=user_id, name="My Watchlist")
        db.add(w)
        db.commit()
        db.refresh(w)
    return w

@router.get("/", response_model=WatchlistResponse)
def get_watchlist(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    w = get_or_create_watchlist(current_user.id, db)
    
    items_output = []
    for item in w.items:
        scheme = db.query(Scheme).filter(Scheme.id == item.scheme_id).first()
        code = scheme.scheme_code if scheme else 119550
        name = scheme.scheme_name if scheme else "Mutual Fund Scheme"
        
        # Get latest NAV & return
        try:
            detail = get_scheme_detail(scheme_code=code, db=db)
            current_nav = detail["current_nav"]
        except Exception:
            current_nav = 100.0

        items_output.append({
            "id": item.id,
            "watchlist_id": w.id,
            "scheme_id": item.scheme_id,
            "scheme_code": code,
            "scheme_name": name,
            "current_nav": current_nav,
            "return_1y": 0.182
        })

    return {
        "id": w.id,
        "name": w.name,
        "created_at": w.created_at,
        "items": items_output
    }

@router.post("/items", response_model=WatchlistItemResponse, status_code=status.HTTP_201_CREATED)
def add_to_watchlist(item_in: WatchlistItemCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    w = get_or_create_watchlist(current_user.id, db)
    
    # Check if item already exists in watchlist
    existing = db.query(WatchlistItem).filter(
        WatchlistItem.watchlist_id == w.id,
        WatchlistItem.scheme_id == item_in.scheme_id
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="Scheme already present in watchlist.")

    item = WatchlistItem(watchlist_id=w.id, scheme_id=item_in.scheme_id)
    db.add(item)
    db.commit()
    db.refresh(item)

    scheme = db.query(Scheme).filter(Scheme.id == item_in.scheme_id).first()
    code = scheme.scheme_code if scheme else 119550
    name = scheme.scheme_name if scheme else "Mutual Fund Scheme"

    return {
        "id": item.id,
        "watchlist_id": w.id,
        "scheme_id": item.scheme_id,
        "scheme_code": code,
        "scheme_name": name,
        "current_nav": 108.73,
        "return_1y": 0.182
    }

@router.delete("/items/{item_id}", status_code=status.HTTP_200_OK)
def remove_from_watchlist(item_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    w = get_or_create_watchlist(current_user.id, db)
    item = db.query(WatchlistItem).filter(WatchlistItem.id == item_id, WatchlistItem.watchlist_id == w.id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Watchlist item not found")
    
    db.delete(item)
    db.commit()
    return {"message": "Item removed from watchlist"}
