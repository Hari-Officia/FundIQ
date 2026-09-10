import os
import sys
from pathlib import Path
import pandas as pd
import datetime
from typing import Dict, Any

# Ensure backend directory is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
FUNDIQ_DIR = SCRIPT_DIR.parent
BACKEND_DIR = FUNDIQ_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.database import engine, Base, SessionLocal
from app.models.scheme import Scheme
from app.models.nav import NAVHistory
from app.models.prediction import Prediction

def determine_category(scheme_name: str, category_id: int = 0) -> str:
    name_lower = str(scheme_name).lower()
    if any(x in name_lower for x in ["equity", "small cap", "mid cap", "large cap", "flexi cap", "multi cap", "index", "elss", "bluechip", "value"]):
        return "Equity"
    elif any(x in name_lower for x in ["debt", "bond", "liquid", "money market", "gilt", "overnight", "credit risk", "corporate", "treasury", "floater"]):
        return "Debt"
    elif any(x in name_lower for x in ["hybrid", "balanced", "arbitrage"]):
        return "Hybrid"
    else:
        if category_id == 1:
            return "Equity"
        elif category_id == 2:
            return "Debt"
        elif category_id == 3:
            return "Hybrid"
        return "Other"

def seed_database():
    print("=" * 60)
    print("      FUNDIQ DATABASE SEEDING & INGESTION PIPELINE         ")
    print("=" * 60)

    # 1. Create DB tables if not present
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    csv_path = FUNDIQ_DIR / "data" / "recommendation_data.csv"
    if not csv_path.exists():
        csv_path = FUNDIQ_DIR / "recommendation_data.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found!")
        return

    print(f"\n[1/3] Ingesting scheme data from {csv_path.name}...")
    df = pd.read_csv(csv_path)

    # Clean missing values
    df['scheme_name'] = df['scheme_name'].fillna("Unknown Scheme")
    df['amc_name'] = df['amc_name'].fillna("General Mutual Fund")
    df['nav'] = df['nav'].fillna(100.0)
    df['predicted_return_5obs'] = df['predicted_return_5obs'].fillna(0.0)

    existing_schemes = {s.scheme_code: s for s in db.query(Scheme).all()}
    print(f" -> Existing schemes in DB: {len(existing_schemes)}")

    new_schemes = []
    nav_entries = []
    pred_entries = []

    today = datetime.date.today()

    for idx, row in df.iterrows():
        code = int(row['scheme_code'])
        name = str(row['scheme_name'])
        amc = str(row['amc_name'])
        cat_id = int(row['category_id']) if 'category_id' in row and pd.notna(row['category_id']) else 0
        category = determine_category(name, cat_id)
        nav_val = float(row['nav'])
        pred_return = float(row['predicted_return_5obs'])

        if code in existing_schemes:
            scheme_obj = existing_schemes[code]
        else:
            scheme_obj = Scheme(
                scheme_code=code,
                scheme_name=name,
                amc_name=amc,
                amc_code=amc.split()[0] if amc else "AMC",
                category=category,
                isin=f"INF{code}K01"
            )
            db.add(scheme_obj)
            db.flush()  # assign scheme_obj.id
            existing_schemes[code] = scheme_obj

        # Add single benchmark NAV history entry for today if none exists
        nav_entries.append(NAVHistory(
            scheme_id=scheme_obj.id,
            date=today,
            nav=nav_val
        ))

        # Add prediction entry
        predicted_nav = round(nav_val * (1.0 + pred_return), 2)
        pred_entries.append(Prediction(
            scheme_id=scheme_obj.id,
            model="xgboost_ensemble",
            prediction_date=today,
            horizon=5,
            predicted_return=round(pred_return, 4),
            predicted_nav=predicted_nav,
            model_version="v1.0"
        ))

    print(f"\n[2/3] Committing {len(existing_schemes)} total schemes to SQLite database...")
    db.commit()

    print(f"\n[3/3] Seeding initial NAV history and XGBoost prediction records...")
    # Seed in chunks to prevent memory overhead
    chunk_size = 5000
    for i in range(0, len(nav_entries), chunk_size):
        db.bulk_save_objects(nav_entries[i:i+chunk_size])
    for i in range(0, len(pred_entries), chunk_size):
        db.bulk_save_objects(pred_entries[i:i+chunk_size])
    db.commit()

    scheme_count = db.query(Scheme).count()
    nav_count = db.query(NAVHistory).count()
    pred_count = db.query(Prediction).count()

    print("\n" + "=" * 60)
    print("SEEDING COMPLETE SUMMARY:")
    print(f" -> Total Schemes in DB: {scheme_count}")
    print(f" -> Total NAV History Rows: {nav_count}")
    print(f" -> Total Predictions Saved: {pred_count}")
    print("=" * 60)
    db.close()

if __name__ == "__main__":
    seed_database()
