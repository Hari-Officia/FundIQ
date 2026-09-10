from sqlalchemy import Column, Integer, Float, Date, ForeignKey, Index
from sqlalchemy.orm import relationship
from app.core.database import Base

class NAVHistory(Base):
    __tablename__ = "nav_history"

    id = Column(Integer, primary_key=True, index=True)
    scheme_id = Column(Integer, ForeignKey("schemes.id", ondelete="CASCADE"), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    nav = Column(Float, nullable=False)

    scheme = relationship("Scheme", back_populates="nav_history")

    __table_args__ = (
        Index("idx_scheme_date", "scheme_id", "date"),
    )
