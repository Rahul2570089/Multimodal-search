from sqlalchemy import Column, Integer, String, Text, DECIMAL, Boolean, DateTime, ARRAY, Float
from sqlalchemy.sql import func
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from .database import Base


class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    price = Column(DECIMAL(10, 2), nullable=False)
    category = Column(String(100), nullable=False, index=True)
    brand = Column(String(100), index=True)
    color = Column(String(50))
    size = Column(String(50))
    material = Column(String(100))
    image_url = Column(String(500))
    thumbnail_url = Column(String(500))
    in_stock = Column(Integer, default=0)
    rating = Column(Float, default=0.00)
    num_reviews = Column(Integer, default=0)
    tags = Column(ARRAY(String))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# Pydantic models for API
class ProductBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    price: float = Field(..., gt=0)
    category: str = Field(..., min_length=1, max_length=100)
    brand: Optional[str] = None
    color: Optional[str] = None
    size: Optional[str] = None
    material: Optional[str] = None
    image_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    in_stock: int = Field(default=0, ge=0)
    rating: Optional[float] = Field(default=0.0, ge=0, le=5)
    num_reviews: Optional[int] = Field(default=0, ge=0)
    tags: Optional[List[str]] = None


class ProductCreate(ProductBase):
    pass


class ProductUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    price: Optional[float] = Field(None, gt=0)
    category: Optional[str] = Field(None, min_length=1, max_length=100)
    brand: Optional[str] = None
    color: Optional[str] = None
    size: Optional[str] = None
    material: Optional[str] = None
    image_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    in_stock: Optional[int] = Field(None, ge=0)
    rating: Optional[float] = Field(None, ge=0, le=5)
    num_reviews: Optional[int] = Field(None, ge=0)
    tags: Optional[List[str]] = None


class ProductResponse(ProductBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ProductSearchResponse(BaseModel):
    products: List[ProductResponse]
    total: int
    page: int
    per_page: int
    total_pages: int
