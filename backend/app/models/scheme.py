from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.orm import relationship
from app.core.database import Base

class Scheme(Base):
    __tablename__ = "schemes"

    id = Column(Integer, primary_key=True, index=True)
    scheme_code = Column(Integer, unique=True, index=True, nullable=False)
    scheme_name = Column(String, index=True, nullable=False)
    isin = Column(String, nullable=True)
    amc_code = Column(String, index=True, nullable=True)
    amc_name = Column(String, index=True, nullable=True)
    category = Column(String, index=True, nullable=True)

    nav_history = relationship("NAVHistory", back_populates="scheme", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="scheme", cascade="all, delete-orphan")
