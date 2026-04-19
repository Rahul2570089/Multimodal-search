from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from models import get_db, SearchLog, SearchLogCreate, SearchLogResponse
from utils import get_product_collection
from typing import List, Optional
import time
import uuid

router = APIRouter()


@router.post("/text")
async def search_text(
    query: str = Form(...),
    category: Optional[str] = Form(None),
    min_price: Optional[float] = Form(None),
    max_price: Optional[float] = Form(None),
    limit: int = Form(20),
    db: Session = Depends(get_db)
):
    """Search products using text query"""
    start_time = time.time()
    
    try:
        # For now, return basic text search results
        # This will be enhanced with RAG in Weekend 2
        from models.product import Product
        from sqlalchemy import or_
        
        db_query = db.query(Product)
        
        # Apply text search
        if query:
            db_query = db_query.filter(
                or_(
                    Product.name.ilike(f"%{query}%"),
                    Product.description.ilike(f"%{query}%")
                )
            )
        
        # Apply filters
        if category:
            db_query = db_query.filter(Product.category.ilike(f"%{category}%"))
        if min_price is not None:
            db_query = db_query.filter(Product.price >= min_price)
        if max_price is not None:
            db_query = db_query.filter(Product.price <= max_price)
        
        products = db_query.limit(limit).all()
        
        response_time = int((time.time() - start_time) * 1000)
        
        # Log search
        search_log = SearchLogCreate(
            query=query,
            query_type="text",
            results_count=len(products),
            response_time_ms=response_time,
            session_id=str(uuid.uuid4())
        )
        
        db_search_log = SearchLog(**search_log.dict())
        db.add(db_search_log)
        db.commit()
        
        return {
            "query": query,
            "results": products,
            "total": len(products),
            "response_time_ms": response_time
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/image")
async def search_image(
    image: UploadFile = File(...),
    category: Optional[str] = Form(None),
    min_price: Optional[float] = Form(None),
    max_price: Optional[float] = Form(None),
    limit: int = Form(20),
    db: Session = Depends(get_db)
):
    """Search products using image similarity"""
    start_time = time.time()
    
    try:
        # For now, return random products as placeholder
        # This will be enhanced with image embeddings in Weekend 3
        from models.product import Product
        import random
        
        products = db.query(Product).limit(limit).all()
        random.shuffle(products)
        
        response_time = int((time.time() - start_time) * 1000)
        
        # Log search
        search_log = SearchLogCreate(
            query=f"image_search_{image.filename}",
            query_type="image",
            results_count=len(products),
            response_time_ms=response_time,
            session_id=str(uuid.uuid4())
        )
        
        db_search_log = SearchLog(**search_log.dict())
        db.add(db_search_log)
        db.commit()
        
        return {
            "query": f"image_search_{image.filename}",
            "results": products,
            "total": len(products),
            "response_time_ms": response_time
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image search failed: {str(e)}")


@router.post("/multimodal")
async def search_multimodal(
    query: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    category: Optional[str] = Form(None),
    min_price: Optional[float] = Form(None),
    max_price: Optional[float] = Form(None),
    limit: int = Form(20),
    db: Session = Depends(get_db)
):
    """Search products using both text and image"""
    start_time = time.time()
    
    if not query and not image:
        raise HTTPException(status_code=400, detail="Either query or image must be provided")
    
    try:
        # For now, combine basic text and image search
        # This will be enhanced with proper multimodal fusion in Weekend 4
        from models.product import Product
        from sqlalchemy import or_
        import random
        
        db_query = db.query(Product)
        
        # Apply text search if provided
        if query:
            db_query = db_query.filter(
                or_(
                    Product.name.ilike(f"%{query}%"),
                    Product.description.ilike(f"%{query}%")
                )
            )
        
        # Apply filters
        if category:
            db_query = db_query.filter(Product.category.ilike(f"%{category}%"))
        if min_price is not None:
            db_query = db_query.filter(Product.price >= min_price)
        if max_price is not None:
            db_query = db_query.filter(Product.price <= max_price)
        
        products = db_query.limit(limit).all()
        
        # If image is provided, shuffle results to simulate image influence
        if image:
            random.shuffle(products)
        
        response_time = int((time.time() - start_time) * 1000)
        
        # Log search
        search_query = f"multimodal_{query or 'no_text'}_{image.filename if image else 'no_image'}"
        search_log = SearchLogCreate(
            query=search_query,
            query_type="multimodal",
            results_count=len(products),
            response_time_ms=response_time,
            session_id=str(uuid.uuid4())
        )
        
        db_search_log = SearchLog(**search_log.dict())
        db.add(db_search_log)
        db.commit()
        
        return {
            "query": search_query,
            "results": products,
            "total": len(products),
            "response_time_ms": response_time
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multimodal search failed: {str(e)}")


@router.get("/analytics")
async def get_search_analytics(db: Session = Depends(get_db)):
    """Get search analytics"""
    try:
        from sqlalchemy import func
        
        # Total searches
        total_searches = db.query(func.count(SearchLog.id)).scalar()
        
        # Average response time
        avg_response_time = db.query(func.avg(SearchLog.response_time_ms)).scalar()
        
        # Search type distribution
        search_types = db.query(
            SearchLog.query_type,
            func.count(SearchLog.id).label('count')
        ).group_by(SearchLog.query_type).all()
        
        # Most common queries
        common_queries = db.query(
            SearchLog.query,
            func.count(SearchLog.id).label('count')
        ).group_by(SearchLog.query).order_by(func.count(SearchLog.id).desc()).limit(10).all()
        
        return {
            "total_searches": total_searches,
            "average_response_time_ms": round(float(avg_response_time), 2) if avg_response_time else 0,
            "search_type_distribution": {st[0]: st[1] for st in search_types},
            "most_common_queries": [{"query": q[0], "count": q[1]} for q in common_queries]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analytics failed: {str(e)}")
