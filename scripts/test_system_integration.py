import os
import sys
from pathlib import Path

# Add backend directory to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
FUNDIQ_DIR = SCRIPT_DIR.parent
BACKEND_DIR = FUNDIQ_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

print("=========================================================================")
print("          FUNDIQ SYSTEM INTEGRATION & CONNECTIONS VERIFICATION           ")
print("=========================================================================")

# 1. Database Connection & Schema Initialization Test
print("\n[1/5] Testing Database Connection & Models...")
try:
    from app.core.database import engine, Base, SessionLocal
    from app.models.user import User, RefreshToken
    from app.models.scheme import Scheme
    from app.models.nav import NAVHistory
    from app.models.portfolio import Watchlist, WatchlistItem, Portfolio, Holding
    from app.models.prediction import Prediction
    from app.models.recommendation import UserProfile, Recommendation, RecommendationItem
    
    # Auto create tables
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    
    # Simple query check
    user_count = db.query(User).count()
    scheme_count = db.query(Scheme).count()
    print(" -> SQLite Database created/connected successfully.")
    print(f" -> DB Query Verification OK (Users: {user_count}, Schemes: {scheme_count})")
    db.close()
    db_status = "PASS"
except Exception as e:
    print(f" -> Database Connection FAILED: {e}")
    db_status = f"FAIL ({e})"

# 2. ML Model Loading & Artifact Registry Test
print("\n[2/5] Testing ML Model Loading & Registry Artifacts...")
try:
    from app.ml.model_loader import model_registry
    model_registry.load_models()
    if model_registry.is_loaded and model_registry.xgb_model is not None:
        print(f" -> Loaded XGBoost Model: {model_registry.xgb_model.__class__.__name__}")
        print(f" -> Loaded Feature Names ({len(model_registry.model_features)} features): {model_registry.model_features}")
        print(f" -> Loaded Model Config Keys: {list(model_registry.recommendation_config.keys())}")
        ml_status = "PASS"
    else:
        print(f" -> ML Model status is_loaded={model_registry.is_loaded}")
        ml_status = "FAIL (Model not loaded)"
except Exception as e:
    print(f" -> ML Model loading FAILED: {e}")
    ml_status = f"FAIL ({e})"

# 3. ML Predictor & Recommendation Engines Test
print("\n[3/5] Testing ML Predictor & Recommendation Engine Logic...")
try:
    from app.ml.predictor import predict_scheme_returns, extract_features_from_nav_history
    from app.ml.recommendation_engine import recommendation_engine
    
    mock_navs = [{"date": "2025-01-01", "nav": 100.0}, {"date": "2025-01-02", "nav": 101.5}, {"date": "2025-01-05", "nav": 102.8}]
    pred_res = predict_scheme_returns(119550, "Aditya Birla Sun Life Banking & PSU Debt Fund", mock_navs, horizon_days=30)
    print(f" -> Predictor Inference OK: Scheme 119550 -> Horizon: {pred_res['horizon_days']}d, Pred Return: {pred_res['predicted_return']*100:.2f}%, Pred NAV: {pred_res['predicted_nav']}")
    
    sample_funds = [
        {"scheme_code": 119550, "scheme_name": "Aditya Birla Sun Life Banking & PSU Debt Fund", "category": "Debt", "vol_20d": 0.005, "ret_20d": 0.015},
        {"scheme_code": 120438, "scheme_name": "Axis Small Cap Fund", "category": "Equity", "vol_20d": 0.025, "ret_20d": 0.045}
    ]
    recs = recommendation_engine.rank_funds(user_risk="Moderate", horizon="3-5 Years", goal="Wealth Accumulation", investment_amount=50000.0, preferred_type="Both", funds_list=sample_funds)
    print(f" -> Recommendation Engine OK: Ranked {len(recs)} funds for Moderate Risk Profile")
    predictor_status = "PASS"
except Exception as e:
    print(f" -> ML Logic FAILED: {e}")
    predictor_status = f"FAIL ({e})"

# 4. FastAPI Backend Routes & Static Frontend Test
print("\n[4/5] Testing Backend REST API Endpoints & Static UI Serving...")
try:
    from fastapi.testclient import TestClient
    from app.main import app
    
    client = TestClient(app)
    
    # 4.1 Health
    r_health = client.get("/health")
    print(f" -> GET /health : {r_health.status_code} ({r_health.json()['status']})")
    
    # 4.2 Schemes List
    r_schemes = client.get("/api/v1/schemes/")
    print(f" -> GET /api/v1/schemes/ : {r_schemes.status_code} (Returned {len(r_schemes.json())} schemes)")
    
    # 4.3 Scheme Detail
    r_scheme_detail = client.get("/api/v1/schemes/119550")
    print(f" -> GET /api/v1/schemes/119550 : {r_scheme_detail.status_code} ({r_scheme_detail.json()['scheme_name']})")
    
    # 4.4 Analytics Metrics
    r_analytics = client.get("/api/v1/analytics/metrics/119550")
    print(f" -> GET /api/v1/analytics/metrics/119550 : {r_analytics.status_code} (1Y Return: {r_analytics.json()['performance']['return_1y']*100:.2f}%)")
    
    # 4.5 Predictions
    r_pred = client.get("/api/v1/predictions/119550?horizon_days=30")
    print(f" -> GET /api/v1/predictions/119550 : {r_pred.status_code} (Model: {r_pred.json()['model']})")
    
    # 4.6 Recommendations POST
    rec_payload = {
        "risk_level": "Aggressive",
        "horizon": "5+ Years",
        "goal": "Wealth Accumulation",
        "investment_amount": 100000.0,
        "preferred_type": "Equity"
    }
    r_rec = client.post("/api/v1/recommendations/generate", json=rec_payload)
    print(f" -> POST /api/v1/recommendations/generate : {r_rec.status_code} (Generated {len(r_rec.json()['recommended_funds'])} recommended funds)")
    
    # 4.7 Web Frontend Static Mounting
    r_ui = client.get("/")
    print(f" -> GET / (Static Web UI) : {r_ui.status_code} (Length: {len(r_ui.text)} bytes)")
    
    if all(res.status_code in (200, 201) for res in [r_health, r_schemes, r_scheme_detail, r_analytics, r_pred, r_rec, r_ui]):
        api_status = "PASS"
    else:
        api_status = "FAIL (One or more HTTP endpoints failed)"
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f" -> Backend API Endpoints FAILED: {e}")
    api_status = f"FAIL ({e})"

# Summary
print("\n================ SYSTEM STATUS SUMMARY ================")
print(f"1. Database (SQLite Database)        : {db_status}")
print(f"2. ML Model Registry & Artifacts     : {ml_status}")
print(f"3. ML Predictor & Recommendation Engine: {predictor_status}")
print(f"4. Backend API Routes & Web UI Mount : {api_status}")
print("=======================================================")
if db_status == "PASS" and ml_status == "PASS" and predictor_status == "PASS" and api_status == "PASS":
    print(">>> ALL CONNECTED & WORKING PROPERLY <<<")
else:
    print(">>> SOME COMPONENTS FAILED VERIFICATION <<<")
