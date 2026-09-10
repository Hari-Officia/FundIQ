from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from app.core.database import Base

class UserProfile(Base):
    __tablename__ = "user_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False)
    risk_level = Column(String, nullable=False, default="Moderate")  # Low, Moderate, High
    horizon = Column(String, nullable=False, default="3-5 years")    # < 1 year, 1-3 years, 3-5 years, > 5 years
    goal = Column(String, nullable=False, default="Wealth growth")   # Capital preservation, Regular income, Wealth growth, Tax saving
    investment_amount = Column(Float, nullable=False, default=10000.0)
    preferred_type = Column(String, nullable=False, default="Equity") # Equity, Debt, Hybrid, No preference
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="profile")

class Recommendation(Base):
    __tablename__ = "recommendations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    generated_at = Column(DateTime, default=datetime.utcnow, index=True)
    risk_level = Column(String, nullable=False)
    horizon = Column(String, nullable=False)
    goal = Column(String, nullable=False)
    investment_amount = Column(Float, nullable=False)
    preferred_type = Column(String, nullable=False)
    model_version = Column(String, default="xgboost_risk_v1")

    items = relationship("RecommendationItem", back_populates="recommendation", cascade="all, delete-orphan", order_by="RecommendationItem.rank")

class RecommendationItem(Base):
    __tablename__ = "recommendation_items"

    id = Column(Integer, primary_key=True, index=True)
    recommendation_id = Column(Integer, ForeignKey("recommendations.id", ondelete="CASCADE"), nullable=False, index=True)
    rank = Column(Integer, nullable=False)  # 1 to 5
    scheme_code = Column(Integer, nullable=False)
    scheme_name = Column(String, nullable=False)
    predicted_return = Column(Float, nullable=False)
    risk_level = Column(String, nullable=False)
    final_score = Column(Float, nullable=False)  # 0 to 100
    reason = Column(Text, nullable=False)

    recommendation = relationship("Recommendation", back_populates="items")
