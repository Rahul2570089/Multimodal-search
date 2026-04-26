"""
Cross-modal retrieval service for text-to-image and image-to-text mapping
"""
import os
import logging
import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json

from .text_embeddings import get_text_embedding_service
from .image_embeddings import get_image_embedding_service
from .rag_chain import get_rag_chain
from .visual_search import get_visual_search_service

logger = logging.getLogger(__name__)

@dataclass
class CrossModalResult:
    """Result from cross-modal retrieval"""
    query_type: str  # 'text_to_image' or 'image_to_text'
    original_query: str
    retrieved_items: List[Dict[str, Any]]
    cross_modal_score: float
    confidence: float
    explanation: Dict[str, Any]

class CrossModalRetrievalService:
    """Service for cross-modal retrieval between text and images"""
    
    def __init__(self):
        self.text_embedding_service = get_text_embedding_service()
        self.image_embedding_service = get_image_embedding_service()
        self.rag_chain = get_rag_chain()
        self.visual_search_service = get_visual_search_service()
        
        # Cross-modal mapping cache
        self.text_to_image_cache = {}
        self.image_to_text_cache = {}
        
        # Cross-modal similarity thresholds
        self.similarity_threshold = 0.3
        self.confidence_threshold = 0.5
        
        # Performance tracking
        self.retrieval_stats = {
            'text_to_image_queries': 0,
            'image_to_text_queries': 0,
            'avg_cross_modal_score': 0.0,
            'cache_hit_rate': 0.0
        }
    
    def text_to_image_retrieval(self, text_query: str, k: int = 10) -> CrossModalResult:
        """Retrieve images based on text query"""
        try:
            start_time = time.time()
            
            # Check cache first
            cache_key = f"{text_query}_{k}"
            if cache_key in self.text_to_image_cache:
                cached_result = self.text_to_image_cache[cache_key]
                self.retrieval_stats['cache_hit_rate'] = (
                    (self.retrieval_stats['cache_hit_rate'] * self.retrieval_stats['text_to_image_queries'] + 1) /
                    (self.retrieval_stats['text_to_image_queries'] + 1)
                )
                return cached_result
            
            # Get text embedding
            text_embedding = self.text_embedding_service.embed_text(text_query)
            
            # Find visually similar products
            visual_results = self.visual_search_service.search_by_image(
                text_embedding, k=k * 2
            )
            
            # Enhance results with text relevance
            enhanced_results = self._enhance_visual_results_with_text(
                visual_results, text_query, text_embedding
            )
            
            # Calculate cross-modal scores
            cross_modal_results = []
            for result in enhanced_results:
                cross_modal_score = self._calculate_text_to_image_score(
                    text_query, result, text_embedding
                )
                
                if cross_modal_score > self.similarity_threshold:
                    result['cross_modal_score'] = cross_modal_score
                    cross_modal_results.append(result)
            
            # Sort by cross-modal score
            cross_modal_results.sort(key=lambda x: x['cross_modal_score'], reverse=True)
            cross_modal_results = cross_modal_results[:k]
            
            # Calculate confidence
            confidence = self._calculate_cross_modal_confidence(cross_modal_results)
            
            # Create result
            result = CrossModalResult(
                query_type='text_to_image',
                original_query=text_query,
                retrieved_items=cross_modal_results,
                cross_modal_score=np.mean([r['cross_modal_score'] for r in cross_modal_results]) if cross_modal_results else 0.0,
                confidence=confidence,
                explanation={
                    'method': 'text_to_image_retrieval',
                    'text_embedding_similarity': self._calculate_text_embedding_similarity(text_query),
                    'visual_matches': len(cross_modal_results),
                    'enhancement_applied': True
                }
            )
            
            # Cache result
            self.text_to_image_cache[cache_key] = result
            
            # Update stats
            self.retrieval_stats['text_to_image_queries'] += 1
            self._update_cross_modal_stats(result.cross_modal_score)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in text-to-image retrieval: {e}")
            return CrossModalResult(
                query_type='text_to_image',
                original_query=text_query,
                retrieved_items=[],
                cross_modal_score=0.0,
                confidence=0.0,
                explanation={'error': str(e)}
            )
    
    def image_to_text_retrieval(self, image_embedding: List[float], k: int = 10) -> CrossModalResult:
        """Retrieve text descriptions based on image embedding"""
        try:
            start_time = time.time()
            
            # Generate text query from image (simplified)
            generated_queries = self._generate_text_queries_from_image(image_embedding)
            
            # Get text search results for generated queries
            all_text_results = []
            for query in generated_queries:
                text_results = self.rag_chain.semantic_search(query, k=k)
                for result in text_results:
                    result['generated_query'] = query
                    all_text_results.append(result)
            
            # Enhance with visual similarity
            enhanced_results = self._enhance_text_results_with_image(
                all_text_results, image_embedding
            )
            
            # Calculate cross-modal scores
            cross_modal_results = []
            for result in enhanced_results:
                cross_modal_score = self._calculate_image_to_text_score(
                    result, image_embedding
                )
                
                if cross_modal_score > self.similarity_threshold:
                    result['cross_modal_score'] = cross_modal_score
                    cross_modal_results.append(result)
            
            # Remove duplicates and sort
            seen_products = set()
            unique_results = []
            for result in cross_modal_results:
                product_id = result.metadata.get('product_id')
                if product_id and product_id not in seen_products:
                    seen_products.add(product_id)
                    unique_results.append(result)
            
            unique_results.sort(key=lambda x: x.metadata.get('cross_modal_score', 0), reverse=True)
            unique_results = unique_results[:k]
            
            # Convert to dict format
            final_results = []
            for result in unique_results:
                final_results.append({
                    'product_id': result.metadata.get('product_id'),
                    'name': result.metadata.get('name', ''),
                    'category': result.metadata.get('category', ''),
                    'brand': result.metadata.get('brand', ''),
                    'price': result.metadata.get('price', 0),
                    'rating': result.metadata.get('rating', 0),
                    'image_url': result.metadata.get('image_url', ''),
                    'cross_modal_score': result.metadata.get('cross_modal_score', 0),
                    'generated_query': result.metadata.get('generated_query', ''),
                    'text_relevance': result.metadata.get('search_score', 0)
                })
            
            # Calculate confidence
            confidence = self._calculate_cross_modal_confidence(final_results)
            
            # Create result
            result = CrossModalResult(
                query_type='image_to_text',
                original_query='image_embedding',
                retrieved_items=final_results,
                cross_modal_score=np.mean([r['cross_modal_score'] for r in final_results]) if final_results else 0.0,
                confidence=confidence,
                explanation={
                    'method': 'image_to_text_retrieval',
                    'generated_queries': generated_queries,
                    'text_matches': len(final_results),
                    'enhancement_applied': True
                }
            )
            
            # Update stats
            self.retrieval_stats['image_to_text_queries'] += 1
            self._update_cross_modal_stats(result.cross_modal_score)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in image-to-text retrieval: {e}")
            return CrossModalResult(
                query_type='image_to_text',
                original_query='image_embedding',
                retrieved_items=[],
                cross_modal_score=0.0,
                confidence=0.0,
                explanation={'error': str(e)}
            )
    
    def _enhance_visual_results_with_text(self, visual_results: List[Dict[str, Any]], 
                                        text_query: str, text_embedding: List[float]) -> List[Dict[str, Any]]:
        """Enhance visual search results with text relevance"""
        try:
            enhanced_results = []
            
            for result in visual_results:
                # Get product details for text matching
                product_text = f"{result.get('name', '')} {result.get('category', '')} {result.get('brand', '')}"
                
                # Calculate text similarity
                product_embedding = self.text_embedding_service.embed_text(product_text)
                text_similarity = self.text_embedding_service.compute_similarity(
                    text_embedding, product_embedding
                )
                
                # Enhance result with text relevance
                enhanced_result = result.copy()
                enhanced_result['text_relevance'] = text_similarity
                enhanced_result['combined_score'] = (
                    result.get('similarity_score', 0) * 0.7 + text_similarity * 0.3
                )
                
                enhanced_results.append(enhanced_result)
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Error enhancing visual results: {e}")
            return visual_results
    
    def _enhance_text_results_with_image(self, text_results: List, 
                                       image_embedding: List[float]) -> List:
        """Enhance text search results with visual similarity"""
        try:
            enhanced_results = []
            
            for result in text_results:
                # Get product image embedding (placeholder)
                product_data = {
                    'id': result.metadata.get('product_id'),
                    'name': result.metadata.get('name', ''),
                    'category': result.metadata.get('category', ''),
                    'brand': result.metadata.get('brand', ''),
                    'image_url': result.metadata.get('image_url', '')
                }
                
                product_image_embedding = self.image_embedding_service.embed_product_image(product_data)
                
                # Calculate visual similarity
                visual_similarity = self.image_embedding_service.compute_similarity(
                    image_embedding, product_image_embedding
                )
                
                # Enhance result with visual relevance
                enhanced_result = result
                enhanced_result.metadata['visual_relevance'] = visual_similarity
                enhanced_result.metadata['cross_modal_score'] = (
                    result.metadata.get('search_score', 0) * 0.6 + visual_similarity * 0.4
                )
                
                enhanced_results.append(enhanced_result)
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Error enhancing text results: {e}")
            return text_results
    
    def _calculate_text_to_image_score(self, text_query: str, visual_result: Dict[str, Any], 
                                      text_embedding: List[float]) -> float:
        """Calculate text-to-image cross-modal score"""
        try:
            # Base scores
            visual_score = visual_result.get('similarity_score', 0)
            text_relevance = visual_result.get('text_relevance', 0)
            
            # Category matching bonus
            text_lower = text_query.lower()
            category_bonus = 0.0
            if visual_result.get('category', '').lower() in text_lower:
                category_bonus = 0.2
            
            # Brand matching bonus
            if visual_result.get('brand', '').lower() in text_lower:
                category_bonus += 0.1
            
            # Combined score
            cross_modal_score = (
                visual_score * 0.5 +
                text_relevance * 0.3 +
                category_bonus
            )
            
            return min(1.0, cross_modal_score)
            
        except Exception as e:
            logger.error(f"Error calculating text-to-image score: {e}")
            return 0.0
    
    def _calculate_image_to_text_score(self, text_result: Any, image_embedding: List[float]) -> float:
        """Calculate image-to-text cross-modal score"""
        try:
            # Base scores
            text_score = text_result.metadata.get('search_score', 0)
            visual_relevance = text_result.metadata.get('visual_relevance', 0)
            
            # Combined score
            cross_modal_score = text_score * 0.6 + visual_relevance * 0.4
            
            return min(1.0, cross_modal_score)
            
        except Exception as e:
            logger.error(f"Error calculating image-to-text score: {e}")
            return 0.0
    
    def _generate_text_queries_from_image(self, image_embedding: List[float]) -> List[str]:
        """Generate text queries from image embedding"""
        try:
            # This is a simplified implementation
            # In production, this would use a trained image-to-text model
            
            # Generate generic queries based on common categories
            base_queries = [
                "product",
                "item",
                "clothing",
                "electronics",
                "accessories"
            ]
            
            # Add attribute-based queries
            attribute_queries = [
                "high quality",
                "popular",
                "best rated",
                "affordable",
                "premium"
            ]
            
            # Combine queries
            generated_queries = []
            for base in base_queries:
                for attr in attribute_queries[:2]:  # Limit to avoid too many queries
                    generated_queries.append(f"{attr} {base}")
            
            return generated_queries[:5]  # Return top 5
            
        except Exception as e:
            logger.error(f"Error generating text queries: {e}")
            return ["product", "item"]
    
    def _calculate_cross_modal_confidence(self, results: List[Dict[str, Any]]) -> float:
        """Calculate confidence in cross-modal retrieval"""
        try:
            if not results:
                return 0.0
            
            # Average cross-modal score
            avg_score = np.mean([r.get('cross_modal_score', 0) for r in results])
            
            # Score consistency (lower variance = higher confidence)
            scores = [r.get('cross_modal_score', 0) for r in results]
            score_variance = np.var(scores)
            consistency_bonus = max(0, 1 - score_variance)
            
            # Result count bonus
            count_bonus = min(0.2, len(results) / 10)
            
            # Final confidence
            confidence = avg_score * 0.6 + consistency_bonus * 0.2 + count_bonus * 0.2
            
            return min(1.0, confidence)
            
        except Exception as e:
            logger.error(f"Error calculating cross-modal confidence: {e}")
            return 0.0
    
    def _calculate_text_embedding_similarity(self, text_query: str) -> float:
        """Calculate text embedding self-similarity (for quality assessment)"""
        try:
            # Generate embedding twice and compare (for consistency check)
            embedding1 = self.text_embedding_service.embed_text(text_query)
            embedding2 = self.text_embedding_service.embed_text(text_query)
            
            similarity = self.text_embedding_service.compute_similarity(embedding1, embedding2)
            return similarity
            
        except Exception as e:
            logger.error(f"Error calculating text embedding similarity: {e}")
            return 0.0
    
    def _update_cross_modal_stats(self, cross_modal_score: float):
        """Update cross-modal retrieval statistics"""
        try:
            total_queries = self.retrieval_stats['text_to_image_queries'] + self.retrieval_stats['image_to_text_queries']
            
            if total_queries > 0:
                self.retrieval_stats['avg_cross_modal_score'] = (
                    (self.retrieval_stats['avg_cross_modal_score'] * (total_queries - 1) + cross_modal_score) /
                    total_queries
                )
            
        except Exception as e:
            logger.error(f"Error updating cross-modal stats: {e}")
    
    def get_cross_modal_analytics(self) -> Dict[str, Any]:
        """Get cross-modal retrieval analytics"""
        try:
            return {
                'text_to_image_queries': self.retrieval_stats['text_to_image_queries'],
                'image_to_text_queries': self.retrieval_stats['image_to_text_queries'],
                'total_queries': self.retrieval_stats['text_to_image_queries'] + self.retrieval_stats['image_to_text_queries'],
                'average_cross_modal_score': round(self.retrieval_stats['avg_cross_modal_score'], 3),
                'cache_hit_rate': round(self.retrieval_stats['cache_hit_rate'], 3),
                'cache_size': len(self.text_to_image_cache) + len(self.image_to_text_cache),
                'performance_metrics': {
                    'avg_score_quality': 'high' if self.retrieval_stats['avg_cross_modal_score'] > 0.7 else 'medium',
                    'cache_efficiency': 'good' if self.retrieval_stats['cache_hit_rate'] > 0.3 else 'needs_improvement'
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting cross-modal analytics: {e}")
            return {}
    
    def clear_cache(self):
        """Clear cross-modal retrieval cache"""
        try:
            self.text_to_image_cache.clear()
            self.image_to_text_cache.clear()
            logger.info("Cross-modal retrieval cache cleared")
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

# Global instance
cross_modal_retrieval_service = CrossModalRetrievalService()

def get_cross_modal_retrieval_service() -> CrossModalRetrievalService:
    """Get global cross-modal retrieval service instance"""
    return cross_modal_retrieval_service
