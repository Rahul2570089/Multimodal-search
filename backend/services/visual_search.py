"""
Visual search service using image embeddings
"""
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

from .image_embeddings import get_image_embedding_service
from .text_embeddings import get_text_embedding_service
from .rag_chain import get_rag_chain
from models.product import Product
from models.database import Base
from utils import get_chroma_client

logger = logging.getLogger(__name__)

class VisualSearchService:
    """Service for visual search and image similarity"""
    
    def __init__(self):
        self.image_embedding_service = get_image_embedding_service()
        self.text_embedding_service = get_text_embedding_service()
        self.rag_chain = get_rag_chain()
        self.chroma_client = get_chroma_client()
        self.collection_name = os.getenv("CHROMA_COLLECTION_NAME", "products")
        self._initialize_image_collection()
    
    def _initialize_image_collection(self):
        """Initialize separate collection for image embeddings"""
        try:
            if self.chroma_client is None:
                logger.error("ChromaDB client not initialized")
                return
            
            # Create image collection if it doesn't exist
            image_collection_name = f"{self.collection_name}_images"
            
            try:
                self.image_collection = self.chroma_client.get_collection(name=image_collection_name)
                logger.info(f"Connected to existing image collection: {image_collection_name}")
            except:
                self.image_collection = self.chroma_client.create_collection(
                    name=image_collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Created new image collection: {image_collection_name}")
                
        except Exception as e:
            logger.error(f"Error initializing image collection: {e}")
            self.image_collection = None
    
    def add_product_image(self, product_data: Dict[str, Any]) -> bool:
        """Add product image to visual search index"""
        try:
            if self.image_collection is None:
                logger.error("Image collection not initialized")
                return False
            
            # Generate image embedding
            image_embedding = self.image_embedding_service.embed_product_image(product_data)
            
            # Create document for image
            image_doc = {
                'id': f"img_{product_data.get('id')}",
                'embedding': image_embedding,
                'metadata': {
                    'product_id': product_data.get('id'),
                    'name': product_data.get('name', ''),
                    'category': product_data.get('category', ''),
                    'brand': product_data.get('brand', ''),
                    'price': product_data.get('price', 0),
                    'rating': product_data.get('rating', 0),
                    'color': product_data.get('color', ''),
                    'material': product_data.get('material', ''),
                    'image_url': product_data.get('image_url', '')
                }
            }
            
            # Add to collection
            self.image_collection.add(
                embeddings=[image_embedding],
                documents=[f"Product: {product_data.get('name')}"],
                metadatas=[image_doc['metadata']],
                ids=[image_doc['id']]
            )
            
            logger.info(f"Added image for product {product_data.get('id')}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding product image: {e}")
            return False
    
    def search_by_image(self, image_embedding: List[float], 
                      k: int = 10,
                      filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar products by image embedding"""
        try:
            if self.image_collection is None:
                logger.error("Image collection not initialized")
                return []
            
            # Build where clause for filters
            where_clause = None
            if filters:
                where_clauses = []
                for key, value in filters.items():
                    if key in ['category', 'brand', 'color']:
                        where_clauses.append({key: {"$eq": value}})
                
                if len(where_clauses) == 1:
                    where_clause = where_clauses[0]
                elif len(where_clauses) > 1:
                    where_clause = {"$and": where_clauses}
            
            # Search in ChromaDB
            results = self.image_collection.query(
                query_embeddings=[image_embedding],
                n_results=k,
                where=where_clause
            )
            
            # Process results
            processed_results = []
            if results['ids'] and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}
                    distance = results['distances'][0][i] if results['distances'] and results['distances'][0] else 1.0
                    
                    # Convert distance to similarity score
                    similarity = 1.0 / (1.0 + distance)
                    
                    processed_results.append({
                        'product_id': metadata.get('product_id'),
                        'name': metadata.get('name', ''),
                        'category': metadata.get('category', ''),
                        'brand': metadata.get('brand', ''),
                        'price': metadata.get('price', 0),
                        'rating': metadata.get('rating', 0),
                        'image_url': metadata.get('image_url', ''),
                        'similarity_score': similarity,
                        'distance': distance,
                        'search_type': 'visual'
                    })
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Error in visual search: {e}")
            return []
    
    def hybrid_visual_search(self, image_embedding: List[float],
                           text_query: Optional[str] = None,
                           k: int = 10,
                           filters: Optional[Dict[str, Any]] = None,
                           image_weight: float = 0.7) -> List[Dict[str, Any]]:
        """Hybrid search combining image and text"""
        try:
            # Visual search results
            visual_results = self.search_by_image(image_embedding, k=k*2, filters=filters)
            
            # Text search results (if query provided)
            text_results = []
            if text_query:
                text_docs = self.rag_chain.semantic_search(text_query, k=k*2, filters=filters)
                text_results = self._convert_text_results(text_docs)
            
            # Combine and re-rank results
            combined_results = self._combine_visual_text_results(
                visual_results, text_results, image_weight
            )
            
            return combined_results[:k]
            
        except Exception as e:
            logger.error(f"Error in hybrid visual search: {e}")
            return self.search_by_image(image_embedding, k=k, filters=filters)
    
    def _convert_text_results(self, text_docs: List) -> List[Dict[str, Any]]:
        """Convert text search results to standard format"""
        try:
            results = []
            for doc in text_docs:
                metadata = doc.metadata
                results.append({
                    'product_id': metadata.get('product_id'),
                    'name': metadata.get('name', ''),
                    'category': metadata.get('category', ''),
                    'brand': metadata.get('brand', ''),
                    'price': metadata.get('price', 0),
                    'rating': metadata.get('rating', 0),
                    'image_url': metadata.get('image_url', ''),
                    'similarity_score': metadata.get('search_score', 0),
                    'search_type': 'text'
                })
            return results
        except Exception as e:
            logger.error(f"Error converting text results: {e}")
            return []
    
    def _combine_visual_text_results(self, visual_results: List[Dict[str, Any]],
                                  text_results: List[Dict[str, Any]],
                                  image_weight: float) -> List[Dict[str, Any]]:
        """Combine visual and text search results"""
        try:
            # Create product lookup
            product_scores = {}
            
            # Add visual results
            for result in visual_results:
                product_id = result['product_id']
                if product_id not in product_scores:
                    product_scores[product_id] = {
                        'product_data': result.copy(),
                        'visual_score': result['similarity_score'],
                        'text_score': 0.0,
                        'combined_score': 0.0
                    }
                else:
                    product_scores[product_id]['visual_score'] = max(
                        product_scores[product_id]['visual_score'],
                        result['similarity_score']
                    )
            
            # Add text results
            for result in text_results:
                product_id = result['product_id']
                if product_id not in product_scores:
                    product_scores[product_id] = {
                        'product_data': result.copy(),
                        'visual_score': 0.0,
                        'text_score': result['similarity_score'],
                        'combined_score': 0.0
                    }
                else:
                    product_scores[product_id]['text_score'] = max(
                        product_scores[product_id]['text_score'],
                        result['similarity_score']
                    )
            
            # Calculate combined scores
            for product_id, scores in product_scores.items():
                combined_score = (
                    scores['visual_score'] * image_weight +
                    scores['text_score'] * (1 - image_weight)
                )
                scores['combined_score'] = combined_score
                scores['product_data']['combined_score'] = combined_score
                scores['product_data']['search_type'] = 'hybrid'
            
            # Sort by combined score
            combined_results = [scores['product_data'] for scores in product_scores.values()]
            combined_results.sort(key=lambda x: x['combined_score'], reverse=True)
            
            return combined_results
            
        except Exception as e:
            logger.error(f"Error combining results: {e}")
            return visual_results
    
    def find_similar_products(self, product_id: int, k: int = 10) -> List[Dict[str, Any]]:
        """Find visually similar products to a given product"""
        try:
            # Get product data
            DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/multimodal_search")
            engine = create_engine(DATABASE_URL)
            SessionLocal = sessionmaker(bind=engine)
            db = SessionLocal()
            
            product = db.query(Product).filter(Product.id == product_id).first()
            if not product:
                db.close()
                return []
            
            # Create product data dict
            product_data = {
                'id': product.id,
                'name': product.name,
                'category': product.category,
                'brand': product.brand,
                'color': product.color,
                'material': product.material,
                'price': float(product.price),
                'rating': float(product.rating),
                'image_url': product.image_url
            }
            
            # Generate image embedding
            image_embedding = self.image_embedding_service.embed_product_image(product_data)
            
            # Search for similar products
            similar_products = self.search_by_image(image_embedding, k=k+1)  # +1 to exclude self
            
            # Exclude the original product
            filtered_results = [
                result for result in similar_products 
                if result['product_id'] != product_id
            ]
            
            db.close()
            return filtered_results[:k]
            
        except Exception as e:
            logger.error(f"Error finding similar products: {e}")
            return []
    
    def get_visual_search_analytics(self) -> Dict[str, Any]:
        """Get analytics for visual search"""
        try:
            # This would typically query search logs for image searches
            # For now, return basic stats
            return {
                'total_image_searches': 0,
                'average_similarity_score': 0.0,
                'most_searched_categories': [],
                'performance_metrics': {
                    'avg_response_time_ms': 0,
                    'cache_hit_rate': 0.0
                }
            }
        except Exception as e:
            logger.error(f"Error getting visual search analytics: {e}")
            return {}
    
    def optimize_image_collection(self):
        """Optimize image collection for better performance"""
        try:
            if self.image_collection is None:
                return
            
            # Get collection stats
            count = self.image_collection.count()
            logger.info(f"Image collection has {count} items")
            
            # In a real implementation, you might:
            # 1. Remove duplicate embeddings
            # 2. Rebuild index for better performance
            # 3. Optimize storage
            
            logger.info("Image collection optimization completed")
            
        except Exception as e:
            logger.error(f"Error optimizing image collection: {e}")

# Global instance
visual_search_service = VisualSearchService()

def get_visual_search_service() -> VisualSearchService:
    """Get global visual search service instance"""
    return visual_search_service
