from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File
from sqlalchemy.orm import Session
from typing import List, Optional
from models import get_db, Product, ProductCreate, ProductUpdate, ProductResponse, ProductSearchResponse
from sqlalchemy import or_, and_, func
import math

router = APIRouter()


@router.post("/", response_model=ProductResponse)
async def create_product(product: ProductCreate, db: Session = Depends(get_db)):
    """Create a new product"""
    db_product = Product(**product.dict())
    db.add(db_product)
    db.commit()
    db.refresh(db_product)
    return db_product


@router.get("/", response_model=ProductSearchResponse)
async def get_products(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    category: Optional[str] = Query(None),
    brand: Optional[str] = Query(None),
    min_price: Optional[float] = Query(None, ge=0),
    max_price: Optional[float] = Query(None, ge=0),
    search: Optional[str] = Query(None),
    sort_by: Optional[str] = Query("created_at", regex="^(name|price|rating|created_at)$"),
    sort_order: Optional[str] = Query("desc", regex="^(asc|desc)$"),
    db: Session = Depends(get_db)
):
    """Get products with filtering, sorting, and pagination"""
    
    # Build query
    query = db.query(Product)
    
    # Apply filters
    if category:
        query = query.filter(Product.category.ilike(f"%{category}%"))
    if brand:
        query = query.filter(Product.brand.ilike(f"%{brand}%"))
    if min_price is not None:
        query = query.filter(Product.price >= min_price)
    if max_price is not None:
        query = query.filter(Product.price <= max_price)
    if search:
        query = query.filter(
            or_(
                Product.name.ilike(f"%{search}%"),
                Product.description.ilike(f"%{search}%"),
                Product.tags.any(search) if Product.tags else Product.name.ilike(f"%{search}%")
            )
        )
    
    # Apply sorting
    sort_column = getattr(Product, sort_by)
    if sort_order == "desc":
        query = query.order_by(sort_column.desc())
    else:
        query = query.order_by(sort_column.asc())
    
    # Get total count
    total = query.count()
    
    # Apply pagination
    offset = (page - 1) * per_page
    products = query.offset(offset).limit(per_page).all()
    
    # Calculate total pages
    total_pages = math.ceil(total / per_page)
    
    return ProductSearchResponse(
        products=products,
        total=total,
        page=page,
        per_page=per_page,
        total_pages=total_pages
    )


@router.get("/{product_id}", response_model=ProductResponse)
async def get_product(product_id: int, db: Session = Depends(get_db)):
    """Get a specific product by ID"""
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return product


@router.put("/{product_id}", response_model=ProductResponse)
async def update_product(
    product_id: int, 
    product_update: ProductUpdate, 
    db: Session = Depends(get_db)
):
    """Update a product"""
    db_product = db.query(Product).filter(Product.id == product_id).first()
    if not db_product:
        raise HTTPException(status_code=404, detail="Product not found")
    
    # Update fields
    update_data = product_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_product, field, value)
    
    db.commit()
    db.refresh(db_product)
    return db_product


@router.delete("/{product_id}")
async def delete_product(product_id: int, db: Session = Depends(get_db)):
    """Delete a product"""
    db_product = db.query(Product).filter(Product.id == product_id).first()
    if not db_product:
        raise HTTPException(status_code=404, detail="Product not found")
    
    db.delete(db_product)
    db.commit()
    return {"message": "Product deleted successfully"}


@router.get("/categories/list")
async def get_categories(db: Session = Depends(get_db)):
    """Get all unique categories"""
    categories = db.query(Product.category).distinct().all()
    return {"categories": [cat[0] for cat in categories]}


@router.get("/brands/list")
async def get_brands(db: Session = Depends(get_db)):
    """Get all unique brands"""
    brands = db.query(Product.brand).filter(Product.brand.isnot(None)).distinct().all()
    return {"brands": [brand[0] for brand in brands]}


@router.get("/stats/summary")
async def get_products_stats(db: Session = Depends(get_db)):
    """Get products statistics"""
    total_products = db.query(func.count(Product.id)).scalar()
    total_categories = db.query(func.count(func.distinct(Product.category))).scalar()
    total_brands = db.query(func.count(func.distinct(Product.brand))).scalar()
    avg_price = db.query(func.avg(Product.price)).scalar()
    avg_rating = db.query(func.avg(Product.rating)).scalar()
    
    return {
        "total_products": total_products,
        "total_categories": total_categories,
        "total_brands": total_brands,
        "average_price": round(float(avg_price), 2) if avg_price else 0,
        "average_rating": round(float(avg_rating), 2) if avg_rating else 0
    }
