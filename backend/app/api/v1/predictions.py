from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional
import json

from app.core.database import get_db
from app.api.v1.schemes import get_scheme_detail
from app.ml.predictor import predict_scheme_returns
from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.models.prediction import Prediction
from app.core.config import settings

router = APIRouter()

# Global in-memory cache fallback if Redis connection is not established
in_memory_cache = {}

def get_cached_prediction(key: str):
    try:
        import redis
        r = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
        cached = r.get(key)
        if cached:
            return json.loads(cached)
    except Exception:
        pass
    return in_memory_cache.get(key)

def set_cached_prediction(key: str, data: dict, expire_seconds: int = 86400):
    try:
        import redis
        r = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
        r.setex(key, expire_seconds, json.dumps(data))
    except Exception:
        pass
    in_memory_cache[key] = data

@router.post("/", response_model=PredictionResponse)
def get_prediction(req: PredictionRequest, db: Session = Depends(get_db)):
    cache_key = f"prediction:{req.scheme_code}:{req.horizon_days}d"
    cached = get_cached_prediction(cache_key)
    if cached:
        return cached

    detail = get_scheme_detail(scheme_code=req.scheme_code, db=db)
    
    # Run prediction inference
    prediction_result = predict_scheme_returns(
        scheme_code=detail["scheme_code"],
        scheme_name=detail["scheme_name"],
        nav_history=detail["nav_history"],
        horizon_days=req.horizon_days
    )

    # Save prediction audit record to DB
    if detail.get("id"):
        try:
            pred_record = Prediction(
                scheme_id=detail["id"],
                model=req.model or "ensemble",
                horizon=req.horizon_days,
                predicted_return=prediction_result["predicted_return"],
                predicted_nav=prediction_result["predicted_nav"],
                model_version=prediction_result["model_version"]
            )
            db.add(pred_record)
            db.commit()
        except Exception:
            db.rollback()

    set_cached_prediction(cache_key, prediction_result)
    return prediction_result

@router.get("/{scheme_code}", response_model=PredictionResponse)
def get_prediction_by_scheme(
    scheme_code: int,
    horizon_days: int = Query(30, description="Prediction horizon: 1, 7, or 30 days"),
    db: Session = Depends(get_db)
):
    req = PredictionRequest(scheme_code=scheme_code, horizon_days=horizon_days)
    return get_prediction(req=req, db=db)
