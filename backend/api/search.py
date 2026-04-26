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
    use_rag: bool = Form(True),
    db: Session = Depends(get_db)
):
    """Search products using text query with RAG enhancement"""
    start_time = time.time()
    
    try:
        from models.product import Product
        from services.rag_chain import get_rag_chain
        from sqlalchemy import or_
        
        results = []
        search_explanation = None
        
        if use_rag:
            # Use RAG for semantic search
            rag_chain = get_rag_chain()
            
            # Build filters for RAG
            filters = {}
            if category:
                filters['category'] = category
            if min_price is not None or max_price is not None:
                # Price filtering will be applied after semantic search
                pass
            
            # Perform semantic search
            rag_results = rag_chain.semantic_search(query, k=limit*2, filters=filters)
            
            # Convert RAG results to Product objects
            product_ids = []
            for doc in rag_results:
                product_id = doc.metadata.get('product_id')
                if product_id and product_id not in product_ids:
                    product_ids.append(product_id)
            
            # Get full product details
            if product_ids:
                products = db.query(Product).filter(Product.id.in_(product_ids)).all()
                
                # Order by RAG results
                product_dict = {p.id: p for p in products}
                for product_id in product_ids:
                    if product_id in product_dict:
                        results.append(product_dict[product_id])
            
            # Get search explanation
            search_explanation = rag_chain.get_search_explanation(query, rag_results)
            
        else:
            # Fallback to basic text search
            db_query = db.query(Product)
            
            if query:
                db_query = db_query.filter(
                    or_(
                        Product.name.ilike(f"%{query}%"),
                        Product.description.ilike(f"%{query}%")
                    )
                )
            
            results = db_query.limit(limit).all()
        
        # Apply additional filters (price range)
        if min_price is not None or max_price is not None:
            filtered_results = []
            for product in results:
                if min_price is not None and product.price < min_price:
                    continue
                if max_price is not None and product.price > max_price:
                    continue
                filtered_results.append(product)
            results = filtered_results
        
        # Limit results
        results = results[:limit]
        
        response_time = int((time.time() - start_time) * 1000)
        
        # Log search
        search_log = SearchLogCreate(
            query=query,
            query_type="text" if not use_rag else "semantic",
            results_count=len(results),
            response_time_ms=response_time,
            session_id=str(uuid.uuid4())
        )
        
        db_search_log = SearchLog(**search_log.dict())
        db.add(db_search_log)
        db.commit()
        
        response_data = {
            "query": query,
            "results": results,
            "total": len(results),
            "response_time_ms": response_time,
            "search_type": "semantic" if use_rag else "keyword"
        }
        
        # Add search explanation if using RAG
        if search_explanation:
            response_data["explanation"] = search_explanation
        
        return response_data
        
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
        from services.visual_search import get_visual_search_service
        from services.image_embeddings import get_image_embedding_service
        from models.product import Product
        
        # Process uploaded image
        visual_search_service = get_visual_search_service()
        image_embedding_service = get_image_embedding_service()
        
        # Read and validate image
        file_content = await image.read()
        
        # Generate image embedding
        image_embedding = image_embedding_service.embed_image(file_content)
        
        # Build filters
        filters = {}
        if category:
            filters['category'] = category
        
        # Perform visual search
        visual_results = visual_search_service.search_by_image(
            image_embedding, k=limit*2, filters=filters
        )
        
        # Convert to Product objects
        product_ids = []
        for result in visual_results:
            product_id = result.get('product_id')
            if product_id and product_id not in product_ids:
                product_ids.append(product_id)
        
        # Get full product details
        products = []
        if product_ids:
            products = db.query(Product).filter(Product.id.in_(product_ids)).all()
            
            # Order by visual search results
            product_dict = {p.id: p for p in products}
            ordered_products = []
            for result in visual_results:
                product_id = result.get('product_id')
                if product_id in product_dict:
                    product = product_dict[product_id]
                    # Add similarity score to product
                    product.similarity_score = result.get('similarity_score', 0)
                    ordered_products.append(product)
            
            products = ordered_products
        
        # Apply price filters
        if min_price is not None or max_price is not None:
            filtered_products = []
            for product in products:
                if min_price is not None and product.price < min_price:
                    continue
                if max_price is not None and product.price > max_price:
                    continue
                filtered_products.append(product)
            products = filtered_products
        
        # Limit results
        products = products[:limit]
        
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
            "response_time_ms": response_time,
            "search_type": "visual",
            "visual_results": visual_results[:len(products)]
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
    image_weight: float = Form(0.7),
    fusion_strategy: str = Form("adaptive_fusion"),
    ranking_context: str = Form("general_search"),
    enable_advanced_ranking: bool = Form(True),
    db: Session = Depends(get_db)
):
    """Advanced multimodal search with fusion and ranking"""
    start_time = time.time()
    
    if not query and not image:
        raise HTTPException(status_code=400, detail="Either query or image must be provided")
    
    try:
        from services.multimodal_fusion import get_multimodal_fusion_service, FusionStrategy
        from services.cross_modal_retrieval import get_cross_modal_retrieval_service
        from services.advanced_ranking import get_advanced_ranking_service, RankingContext
        from services.fusion_analytics import get_fusion_analytics_service
        from services.image_embeddings import get_image_embedding_service
        from models.product import Product
        
        # Initialize services
        fusion_service = get_multimodal_fusion_service()
        cross_modal_service = get_cross_modal_retrieval_service()
        ranking_service = get_advanced_ranking_service()
        analytics_service = get_fusion_analytics_service()
        image_embedding_service = get_image_embedding_service()
        
        # Parse strategy and context
        try:
            strategy = FusionStrategy(fusion_strategy)
        except ValueError:
            strategy = FusionStrategy.ADAPTIVE_FUSION
        
        try:
            context = RankingContext(ranking_context)
        except ValueError:
            context = RankingContext.GENERAL_SEARCH
        
        # Build filters
        filters = {}
        if category:
            filters['category'] = category
        
        # Get image embedding if provided
        image_embedding = None
        if image:
            file_content = await image.read()
            image_embedding = image_embedding_service.embed_image(file_content)
        
        # Perform fusion search
        fusion_results = fusion_service.fuse_search_results(
            text_query=query,
            image_embedding=image_embedding,
            filters=filters,
            strategy=strategy,
            top_k=limit * 2
        )
        
        # Apply advanced ranking if enabled
        if enable_advanced_ranking and fusion_results:
            # Build user context (simplified)
            user_context = {
                'preferred_categories': [category] if category else [],
                'preferred_brands': [],
                'interaction_history': [],
                'target_category': category
            }
            
            ranked_results = ranking_service.rank_search_results(
                fusion_results, context, user_context, limit
            )
        else:
            # Convert fusion results to simple format
            ranked_results = []
            for result in fusion_results[:limit]:
                ranked_results.append({
                    'product_id': result.product_id,
                    'name': result.name,
                    'category': result.category,
                    'brand': result.brand,
                    'price': result.price,
                    'rating': result.rating,
                    'image_url': result.image_url,
                    'fusion_score': result.fusion_score,
                    'confidence': result.confidence,
                    'strategy_used': result.strategy_used
                })
        
        # Get full product details
        product_ids = [r['product_id'] for r in ranked_results if 'product_id' in r]
        products = []
        if product_ids:
            db_products = db.query(Product).filter(Product.id.in_(product_ids)).all()
            product_dict = {p.id: p for p in db_products}
            
            for ranked_result in ranked_results:
                product_id = ranked_result.get('product_id')
                if product_id in product_dict:
                    product = product_dict[product_id]
                    
                    # Add ranking information
                    if 'fusion_score' in ranked_result:
                        product.fusion_score = ranked_result['fusion_score']
                    if 'confidence' in ranked_result:
                        product.confidence = ranked_result['confidence']
                    if 'strategy_used' in ranked_result:
                        product.strategy_used = ranked_result['strategy_used']
                    if 'final_score' in ranked_result:
                        product.final_score = ranked_result['final_score']
                    if 'rank_position' in ranked_result:
                        product.rank_position = ranked_result['rank_position']
                    
                    products.append(product)
        
        # Apply price filters
        if min_price is not None or max_price is not None:
            filtered_products = []
            for product in products:
                if min_price is not None and product.price < min_price:
                    continue
                if max_price is not None and product.price > max_price:
                    continue
                filtered_products.append(product)
            products = filtered_products
        
        # Limit results
        products = products[:limit]
        
        response_time = int((time.time() - start_time) * 1000)
        
        # Record analytics
        if fusion_results:
            avg_fusion_score = sum(r.fusion_score for r in fusion_results) / len(fusion_results)
            avg_confidence = sum(r.confidence for r in fusion_results) / len(fusion_results)
            
            analytics_service.record_fusion_metric(
                strategy=strategy.value,
                execution_time_ms=response_time,
                fusion_score=avg_fusion_score,
                confidence=avg_confidence,
                result_count=len(products),
                text_available=query is not None,
                image_available=image_embedding is not None
            )
        
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
        
        # Prepare response
        response_data = {
            "query": search_query,
            "results": products,
            "total": len(products),
            "response_time_ms": response_time,
            "search_type": "advanced_multimodal",
            "fusion_strategy": strategy.value,
            "ranking_context": context.value,
            "advanced_ranking_enabled": enable_advanced_ranking,
            "image_weight": image_weight if image else 0
        }
        
        # Add fusion analytics if available
        if fusion_results:
            response_data["fusion_analytics"] = {
                "avg_fusion_score": avg_fusion_score,
                "avg_confidence": avg_confidence,
                "strategy_distribution": fusion_service.get_fusion_analytics()
            }
        
        return response_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Advanced multimodal search failed: {str(e)}")


@router.post("/understand")
async def understand_query(
    query: str = Form(...),
    db: Session = Depends(get_db)
):
    """Analyze and understand search query with intent extraction"""
    try:
        from services.document_processor import get_document_processor
        
        processor = get_document_processor()
        
        # Extract intent
        intent = processor.extract_search_intent(query)
        
        # Expand query
        expanded_queries = processor.expand_query(query)
        
        return {
            "original_query": query,
            "intent": intent,
            "expanded_queries": expanded_queries,
            "suggestions": _generate_query_suggestions(intent, query)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query understanding failed: {str(e)}")


def _generate_query_suggestions(intent: dict, original_query: str) -> List[str]:
    """Generate query suggestions based on intent"""
    suggestions = []
    
    # Category-based suggestions
    if intent.get('category'):
        category = intent['category']
        suggestions.extend([
            f"best {category} products",
            f"affordable {category}",
            f"premium {category}"
        ])
    
    # Brand-based suggestions
    if intent.get('brand'):
        brand = intent['brand']
        suggestions.extend([
            f"{brand} products",
            f"cheap {brand}",
            f"new {brand} items"
        ])
    
    # Price-based suggestions
    if intent.get('price_range') == 'low':
        suggestions.extend([
            "budget-friendly options",
            "affordable alternatives",
            "best value products"
        ])
    elif intent.get('price_range') == 'high':
        suggestions.extend([
            "luxury products",
            "premium items",
            "high-end options"
        ])
    
    # Color-based suggestions
    if intent.get('color'):
        color = intent['color']
        suggestions.extend([
            f"{color} products",
            f"best {color} items",
            f"{color} accessories"
        ])
    
    return list(set(suggestions))[:5]  # Remove duplicates and limit to 5


@router.post("/cross-modal/text-to-image")
async def text_to_image_search(
    query: str = Form(...),
    category: Optional[str] = Form(None),
    limit: int = Form(10),
    db: Session = Depends(get_db)
):
    """Cross-modal search from text to images"""
    start_time = time.time()
    
    try:
        from services.cross_modal_retrieval import get_cross_modal_retrieval_service
        from services.fusion_analytics import get_fusion_analytics_service
        from models.product import Product
        
        cross_modal_service = get_cross_modal_retrieval_service()
        analytics_service = get_fusion_analytics_service()
        
        # Perform cross-modal search
        result = cross_modal_service.text_to_image_retrieval(query, limit)
        
        # Convert to Product objects
        product_ids = [item['product_id'] for item in result.retrieved_items]
        products = []
        if product_ids:
            db_products = db.query(Product).filter(Product.id.in_(product_ids)).all()
            product_dict = {p.id: p for p in db_products}
            
            for item in result.retrieved_items:
                product_id = item['product_id']
                if product_id in product_dict:
                    product = product_dict[product_id]
                    product.cross_modal_score = item['cross_modal_score']
                    products.append(product)
        
        response_time = int((time.time() - start_time) * 1000)
        
        # Record analytics
        analytics_service.record_cross_modal_metric(
            query_type='text_to_image',
            execution_time_ms=response_time,
            cross_modal_score=result.cross_modal_score,
            confidence=result.confidence,
            result_count=len(products)
        )
        
        return {
            "query": query,
            "results": products,
            "total": len(products),
            "response_time_ms": response_time,
            "cross_modal_score": result.cross_modal_score,
            "confidence": result.confidence,
            "explanation": result.explanation
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cross-modal search failed: {str(e)}")


@router.post("/cross-modal/image-to-text")
async def image_to_text_search(
    image: UploadFile = File(...),
    limit: int = Form(10),
    db: Session = Depends(get_db)
):
    """Cross-modal search from image to text"""
    start_time = time.time()
    
    try:
        from services.cross_modal_retrieval import get_cross_modal_retrieval_service
        from services.fusion_analytics import get_fusion_analytics_service
        from services.image_embeddings import get_image_embedding_service
        from models.product import Product
        
        cross_modal_service = get_cross_modal_retrieval_service()
        analytics_service = get_fusion_analytics_service()
        image_embedding_service = get_image_embedding_service()
        
        # Get image embedding
        file_content = await image.read()
        image_embedding = image_embedding_service.embed_image(file_content)
        
        # Perform cross-modal search
        result = cross_modal_service.image_to_text_retrieval(image_embedding, limit)
        
        # Convert to Product objects
        product_ids = [item['product_id'] for item in result.retrieved_items]
        products = []
        if product_ids:
            db_products = db.query(Product).filter(Product.id.in_(product_ids)).all()
            product_dict = {p.id: p for p in db_products}
            
            for item in result.retrieved_items:
                product_id = item['product_id']
                if product_id in product_dict:
                    product = product_dict[product_id]
                    product.cross_modal_score = item['cross_modal_score']
                    product.generated_query = item['generated_query']
                    products.append(product)
        
        response_time = int((time.time() - start_time) * 1000)
        
        # Record analytics
        analytics_service.record_cross_modal_metric(
            query_type='image_to_text',
            execution_time_ms=response_time,
            cross_modal_score=result.cross_modal_score,
            confidence=result.confidence,
            result_count=len(products)
        )
        
        return {
            "query": f"image_search_{image.filename}",
            "results": products,
            "total": len(products),
            "response_time_ms": response_time,
            "cross_modal_score": result.cross_modal_score,
            "confidence": result.confidence,
            "explanation": result.explanation
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cross-modal search failed: {str(e)}")


@router.get("/fusion/analytics")
async def get_fusion_analytics(time_range_hours: int = 24):
    """Get fusion analytics and performance metrics"""
    try:
        from services.fusion_analytics import get_fusion_analytics_service
        
        analytics_service = get_fusion_analytics_service()
        analytics = analytics_service.get_fusion_analytics(time_range_hours)
        
        return analytics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fusion analytics failed: {str(e)}")


@router.get("/fusion/comprehensive")
async def get_comprehensive_analytics(time_range_hours: int = 24):
    """Get comprehensive analytics across all services"""
    try:
        from services.fusion_analytics import get_fusion_analytics_service
        
        analytics_service = get_fusion_analytics_service()
        analytics = analytics_service.get_comprehensive_analytics(time_range_hours)
        
        return analytics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comprehensive analytics failed: {str(e)}")


@router.get("/performance/cache-stats")
async def get_cache_stats():
    """Get cache performance statistics"""
    try:
        from services.performance_optimizer import get_performance_optimizer
        
        optimizer = get_performance_optimizer()
        cache_stats = optimizer.get_cache_stats()
        
        return cache_stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache stats failed: {str(e)}")


@router.get("/performance/metrics")
async def get_performance_metrics():
    """Get performance metrics"""
    try:
        from services.performance_optimizer import get_performance_optimizer
        
        optimizer = get_performance_optimizer()
        perf_stats = optimizer.get_performance_stats()
        
        return perf_stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Performance metrics failed: {str(e)}")


@router.post("/performance/cache/clear")
async def clear_cache():
    """Clear performance cache"""
    try:
        from services.performance_optimizer import get_performance_optimizer
        
        optimizer = get_performance_optimizer()
        optimizer.clear_cache()
        
        return {"message": "Cache cleared successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache clear failed: {str(e)}")


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
