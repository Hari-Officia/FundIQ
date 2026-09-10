from sqlalchemy import Column, Integer, String, Float, Date, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from app.core.database import Base

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    scheme_id = Column(Integer, ForeignKey("schemes.id", ondelete="CASCADE"), nullable=False)
    model = Column(String, nullable=False)  # 'random_forest', 'xgboost', 'ensemble'
    prediction_date = Column(Date, nullable=False, default=datetime.utcnow)
    horizon = Column(Integer, nullable=False, default=30)  # horizon in days: 1, 7, 30
    predicted_return = Column(Float, nullable=False)
    predicted_nav = Column(Float, nullable=False)
    model_version = Column(String, default="v1.0")
    created_at = Column(DateTime, default=datetime.utcnow)

    scheme = relationship("Scheme", back_populates="predictions")
