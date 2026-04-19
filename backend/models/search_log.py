from sqlalchemy import Column, Integer, String, Text, DECIMAL, DateTime, INET, Float
from sqlalchemy.sql import func
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
from .database import Base


class SearchLog(Base):
    __tablename__ = "search_logs"

    id = Column(Integer, primary_key=True, index=True)
    query = Column(Text, nullable=False)
    query_type = Column(String(20), nullable=False, index=True)  # 'text', 'image', 'multimodal'
    results_count = Column(Integer)
    click_through_rate = Column(DECIMAL(5, 4))
    response_time_ms = Column(Integer)
    user_id = Column(String(100))
    session_id = Column(String(100))
    ip_address = Column(INET)
    user_agent = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)


# Pydantic models for API
class SearchLogBase(BaseModel):
    query: str = Field(..., min_length=1)
    query_type: str = Field(..., regex="^(text|image|multimodal)$")
    results_count: Optional[int] = Field(None, ge=0)
    click_through_rate: Optional[float] = Field(None, ge=0, le=1)
    response_time_ms: Optional[int] = Field(None, ge=0)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class SearchLogCreate(SearchLogBase):
    pass


class SearchLogResponse(SearchLogBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class SearchAnalytics(BaseModel):
    total_searches: int
    average_response_time: float
    most_common_queries: list
    search_type_distribution: dict
    daily_search_counts: list
