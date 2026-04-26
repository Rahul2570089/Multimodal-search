"""
Advanced multimodal fusion algorithms for combining text and image search
"""
import os
import logging
import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import json

from .text_embeddings import get_text_embedding_service
from .image_embeddings import get_image_embedding_service
from .rag_chain import get_rag_chain
from .visual_search import get_visual_search_service

logger = logging.getLogger(__name__)

class FusionStrategy(Enum):
    """Fusion strategy types"""
    WEIGHTED_AVERAGE = "weighted_average"
    ADAPTIVE_FUSION = "adaptive_fusion"
    CROSS_MODAL = "cross_modal"
    NEURAL_FUSION = "neural_fusion"
    ENSEMBLE_FUSION = "ensemble_fusion"

@dataclass
class FusionResult:
    """Result from fusion process"""
    product_id: int
    name: str
    category: str
    brand: str
    price: float
    rating: float
    image_url: str
    text_score: float
    image_score: float
    fusion_score: float
    strategy_used: str
    confidence: float
    explanation: Dict[str, Any]

class MultimodalFusionService:
    """Service for advanced multimodal fusion"""
    
    def __init__(self):
        self.text_embedding_service = get_text_embedding_service()
        self.image_embedding_service = get_image_embedding_service()
        self.rag_chain = get_rag_chain()
        self.visual_search_service = get_visual_search_service()
        
        # Fusion parameters
        self.default_text_weight = 0.4
        self.default_image_weight = 0.6
        self.adaptive_threshold = 0.7
        self.confidence_threshold = 0.5
        
        # Performance tracking
        self.fusion_stats = {
            'total_fusions': 0,
            'strategy_usage': {strategy.value: 0 for strategy in FusionStrategy},
            'avg_confidence': 0.0,
            'avg_fusion_time': 0.0
        }
    
    def fuse_search_results(self, 
                          text_query: Optional[str] = None,
                          image_embedding: Optional[List[float]] = None,
                          filters: Optional[Dict[str, Any]] = None,
                          strategy: FusionStrategy = FusionStrategy.ADAPTIVE_FUSION,
                          top_k: int = 10) -> List[FusionResult]:
        """Fuse text and image search results using specified strategy"""
        try:
            start_time = time.time()
            
            # Get individual search results
            text_results = []
            image_results = []
            
            if text_query:
                text_results = self._get_text_search_results(text_query, filters, top_k * 2)
            
            if image_embedding:
                image_results = self._get_image_search_results(image_embedding, filters, top_k * 2)
            
            # Apply fusion strategy
            if strategy == FusionStrategy.WEIGHTED_AVERAGE:
                fused_results = self._weighted_average_fusion(text_results, image_results, top_k)
            elif strategy == FusionStrategy.ADAPTIVE_FUSION:
                fused_results = self._adaptive_fusion(text_results, image_results, top_k)
            elif strategy == FusionStrategy.CROSS_MODAL:
                fused_results = self._cross_modal_fusion(text_query, image_embedding, text_results, image_results, top_k)
            elif strategy == FusionStrategy.NEURAL_FUSION:
                fused_results = self._neural_fusion(text_results, image_results, top_k)
            elif strategy == FusionStrategy.ENSEMBLE_FUSION:
                fused_results = self._ensemble_fusion(text_results, image_results, top_k)
            else:
                fused_results = self._adaptive_fusion(text_results, image_results, top_k)
            
            # Update stats
            fusion_time = time.time() - start_time
            self._update_fusion_stats(strategy, fused_results, fusion_time)
            
            return fused_results
            
        except Exception as e:
            logger.error(f"Error in fusion search: {e}")
            return []
    
    def _get_text_search_results(self, query: str, filters: Optional[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        """Get text search results"""
        try:
            docs = self.rag_chain.semantic_search(query, k=k, filters=filters)
            results = []
            
            for doc in docs:
                metadata = doc.metadata
                results.append({
                    'product_id': metadata.get('product_id'),
                    'name': metadata.get('name', ''),
                    'category': metadata.get('category', ''),
                    'brand': metadata.get('brand', ''),
                    'price': metadata.get('price', 0),
                    'rating': metadata.get('rating', 0),
                    'image_url': metadata.get('image_url', ''),
                    'score': metadata.get('search_score', 0),
                    'semantic_score': metadata.get('semantic_score', 0),
                    'search_type': 'text'
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting text search results: {e}")
            return []
    
    def _get_image_search_results(self, embedding: List[float], filters: Optional[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        """Get image search results"""
        try:
            results = self.visual_search_service.search_by_image(embedding, k=k, filters=filters)
            
            for result in results:
                result['search_type'] = 'image'
                result['score'] = result.get('similarity_score', 0)
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting image search results: {e}")
            return []
    
    def _weighted_average_fusion(self, text_results: List[Dict[str, Any]], 
                                image_results: List[Dict[str, Any]], 
                                top_k: int) -> List[FusionResult]:
        """Simple weighted average fusion"""
        try:
            # Create product lookup
            product_scores = {}
            
            # Process text results
            for result in text_results:
                product_id = result['product_id']
                if product_id not in product_scores:
                    product_scores[product_id] = {
                        'data': result,
                        'text_score': result['score'],
                        'image_score': 0.0
                    }
                else:
                    product_scores[product_id]['text_score'] = max(
                        product_scores[product_id]['text_score'],
                        result['score']
                    )
            
            # Process image results
            for result in image_results:
                product_id = result['product_id']
                if product_id not in product_scores:
                    product_scores[product_id] = {
                        'data': result,
                        'text_score': 0.0,
                        'image_score': result['score']
                    }
                else:
                    product_scores[product_id]['image_score'] = max(
                        product_scores[product_id]['image_score'],
                        result['score']
                    )
            
            # Calculate fusion scores
            fusion_results = []
            for product_id, scores in product_scores.items():
                data = scores['data']
                fusion_score = (
                    scores['text_score'] * self.default_text_weight +
                    scores['image_score'] * self.default_image_weight
                )
                
                confidence = self._calculate_confidence(scores['text_score'], scores['image_score'])
                
                fusion_result = FusionResult(
                    product_id=product_id,
                    name=data['name'],
                    category=data['category'],
                    brand=data['brand'],
                    price=data['price'],
                    rating=data['rating'],
                    image_url=data['image_url'],
                    text_score=scores['text_score'],
                    image_score=scores['image_score'],
                    fusion_score=fusion_score,
                    strategy_used=FusionStrategy.WEIGHTED_AVERAGE.value,
                    confidence=confidence,
                    explanation={
                        'method': 'weighted_average',
                        'text_weight': self.default_text_weight,
                        'image_weight': self.default_image_weight,
                        'has_text': scores['text_score'] > 0,
                        'has_image': scores['image_score'] > 0
                    }
                )
                fusion_results.append(fusion_result)
            
            # Sort by fusion score
            fusion_results.sort(key=lambda x: x.fusion_score, reverse=True)
            return fusion_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in weighted average fusion: {e}")
            return []
    
    def _adaptive_fusion(self, text_results: List[Dict[str, Any]], 
                       image_results: List[Dict[str, Any]], 
                       top_k: int) -> List[FusionResult]:
        """Adaptive fusion based on result quality"""
        try:
            # Analyze result quality
            text_quality = self._analyze_result_quality(text_results)
            image_quality = self._analyze_result_quality(image_results)
            
            # Adapt weights based on quality
            total_quality = text_quality + image_quality
            if total_quality > 0:
                text_weight = text_quality / total_quality
                image_weight = image_quality / total_quality
            else:
                text_weight = self.default_text_weight
                image_weight = self.default_image_weight
            
            # Create product lookup
            product_scores = {}
            
            # Process results (same as weighted average but with adaptive weights)
            for result in text_results:
                product_id = result['product_id']
                if product_id not in product_scores:
                    product_scores[product_id] = {
                        'data': result,
                        'text_score': result['score'],
                        'image_score': 0.0
                    }
                else:
                    product_scores[product_id]['text_score'] = max(
                        product_scores[product_id]['text_score'],
                        result['score']
                    )
            
            for result in image_results:
                product_id = result['product_id']
                if product_id not in product_scores:
                    product_scores[product_id] = {
                        'data': result,
                        'text_score': 0.0,
                        'image_score': result['score']
                    }
                else:
                    product_scores[product_id]['image_score'] = max(
                        product_scores[product_id]['image_score'],
                        result['score']
                    )
            
            # Calculate fusion scores with adaptive weights
            fusion_results = []
            for product_id, scores in product_scores.items():
                data = scores['data']
                fusion_score = (
                    scores['text_score'] * text_weight +
                    scores['image_score'] * image_weight
                )
                
                confidence = self._calculate_confidence(scores['text_score'], scores['image_score'])
                
                fusion_result = FusionResult(
                    product_id=product_id,
                    name=data['name'],
                    category=data['category'],
                    brand=data['brand'],
                    price=data['price'],
                    rating=data['rating'],
                    image_url=data['image_url'],
                    text_score=scores['text_score'],
                    image_score=scores['image_score'],
                    fusion_score=fusion_score,
                    strategy_used=FusionStrategy.ADAPTIVE_FUSION.value,
                    confidence=confidence,
                    explanation={
                        'method': 'adaptive_fusion',
                        'text_weight': text_weight,
                        'image_weight': image_weight,
                        'text_quality': text_quality,
                        'image_quality': image_quality,
                        'has_text': scores['text_score'] > 0,
                        'has_image': scores['image_score'] > 0
                    }
                )
                fusion_results.append(fusion_result)
            
            # Sort by fusion score
            fusion_results.sort(key=lambda x: x.fusion_score, reverse=True)
            return fusion_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in adaptive fusion: {e}")
            return []
    
    def _cross_modal_fusion(self, text_query: Optional[str], 
                          image_embedding: Optional[List[float]],
                          text_results: List[Dict[str, Any]], 
                          image_results: List[Dict[str, Any]], 
                          top_k: int) -> List[FusionResult]:
        """Cross-modal fusion using text-to-image and image-to-text mapping"""
        try:
            fusion_results = []
            
            # If both text and image are available, perform cross-modal enhancement
            if text_query and image_embedding:
                # Enhance text results with visual similarity
                for text_result in text_results:
                    # Find visual similarity for text results
                    best_visual_match = self._find_best_visual_match(text_result, image_results)
                    
                    if best_visual_match:
                        # Cross-modal enhancement
                        cross_modal_score = self._calculate_cross_modal_score(
                            text_result, best_visual_match
                        )
                        
                        confidence = self._calculate_cross_modal_confidence(
                            text_result, best_visual_match
                        )
                        
                        fusion_result = FusionResult(
                            product_id=text_result['product_id'],
                            name=text_result['name'],
                            category=text_result['category'],
                            brand=text_result['brand'],
                            price=text_result['price'],
                            rating=text_result['rating'],
                            image_url=text_result['image_url'],
                            text_score=text_result['score'],
                            image_score=best_visual_match['score'],
                            fusion_score=cross_modal_score,
                            strategy_used=FusionStrategy.CROSS_MODAL.value,
                            confidence=confidence,
                            explanation={
                                'method': 'cross_modal',
                                'cross_modal_boost': cross_modal_score - max(text_result['score'], best_visual_match['score']),
                                'visual_similarity': best_visual_match['score'],
                                'text_relevance': text_result['score']
                            }
                        )
                        fusion_results.append(fusion_result)
                
                # Add image-only results that don't have text matches
                text_product_ids = {r['product_id'] for r in text_results}
                for image_result in image_results:
                    if image_result['product_id'] not in text_product_ids:
                        fusion_result = FusionResult(
                            product_id=image_result['product_id'],
                            name=image_result['name'],
                            category=image_result['category'],
                            brand=image_result['brand'],
                            price=image_result['price'],
                            rating=image_result['rating'],
                            image_url=image_result['image_url'],
                            text_score=0.0,
                            image_score=image_result['score'],
                            fusion_score=image_result['score'] * 0.8,  # Discount for no text match
                            strategy_used=FusionStrategy.CROSS_MODAL.value,
                            confidence=0.6,
                            explanation={
                                'method': 'cross_modal',
                                'reason': 'image_only_result',
                                'visual_similarity': image_result['score']
                            }
                        )
                        fusion_results.append(fusion_result)
            
            else:
                # Fallback to adaptive fusion
                return self._adaptive_fusion(text_results, image_results, top_k)
            
            # Sort by fusion score
            fusion_results.sort(key=lambda x: x.fusion_score, reverse=True)
            return fusion_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in cross-modal fusion: {e}")
            return []
    
    def _neural_fusion(self, text_results: List[Dict[str, Any]], 
                      image_results: List[Dict[str, Any]], 
                      top_k: int) -> List[FusionResult]:
        """Neural network-based fusion (simplified implementation)"""
        try:
            # For this implementation, we'll use a learned combination of features
            # In a production system, this would use a trained neural network
            
            fusion_results = []
            
            # Create product lookup
            product_scores = {}
            
            # Process text results
            for result in text_results:
                product_id = result['product_id']
                if product_id not in product_scores:
                    product_scores[product_id] = {
                        'data': result,
                        'text_score': result['score'],
                        'image_score': 0.0,
                        'features': self._extract_text_features(result)
                    }
                else:
                    product_scores[product_id]['text_score'] = max(
                        product_scores[product_id]['text_score'],
                        result['score']
                    )
            
            # Process image results
            for result in image_results:
                product_id = result['product_id']
                if product_id not in product_scores:
                    product_scores[product_id] = {
                        'data': result,
                        'text_score': 0.0,
                        'image_score': result['score'],
                        'features': self._extract_image_features(result)
                    }
                else:
                    product_scores[product_id]['image_score'] = max(
                        product_scores[product_id]['image_score'],
                        result['score']
                    )
                    # Merge features
                    image_features = self._extract_image_features(result)
                    product_scores[product_id]['features'].update(image_features)
            
            # Apply neural fusion (simplified as weighted combination with learned weights)
            for product_id, scores in product_scores.items():
                data = scores['data']
                features = scores['features']
                
                # Simulated neural network computation
                neural_score = self._simulate_neural_fusion(
                    scores['text_score'], 
                    scores['image_score'], 
                    features
                )
                
                confidence = min(0.9, neural_score + 0.1)  # Boost confidence for neural fusion
                
                fusion_result = FusionResult(
                    product_id=product_id,
                    name=data['name'],
                    category=data['category'],
                    brand=data['brand'],
                    price=data['price'],
                    rating=data['rating'],
                    image_url=data['image_url'],
                    text_score=scores['text_score'],
                    image_score=scores['image_score'],
                    fusion_score=neural_score,
                    strategy_used=FusionStrategy.NEURAL_FUSION.value,
                    confidence=confidence,
                    explanation={
                        'method': 'neural_fusion',
                        'features_used': list(features.keys()),
                        'neural_confidence': confidence,
                        'has_text': scores['text_score'] > 0,
                        'has_image': scores['image_score'] > 0
                    }
                )
                fusion_results.append(fusion_result)
            
            # Sort by fusion score
            fusion_results.sort(key=lambda x: x.fusion_score, reverse=True)
            return fusion_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in neural fusion: {e}")
            return []
    
    def _ensemble_fusion(self, text_results: List[Dict[str, Any]], 
                        image_results: List[Dict[str, Any]], 
                        top_k: int) -> List[FusionResult]:
        """Ensemble fusion combining multiple strategies"""
        try:
            # Get results from multiple strategies
            weighted_results = self._weighted_average_fusion(text_results, image_results, top_k * 2)
            adaptive_results = self._adaptive_fusion(text_results, image_results, top_k * 2)
            cross_modal_results = self._cross_modal_fusion(
                None, None, text_results, image_results, top_k * 2
            )
            
            # Combine ensemble scores
            product_ensemble_scores = {}
            
            # Process results from each strategy
            for strategy_results in [weighted_results, adaptive_results, cross_modal_results]:
                for result in strategy_results:
                    product_id = result.product_id
                    if product_id not in product_ensemble_scores:
                        product_ensemble_scores[product_id] = {
                            'data': result,
                            'scores': [],
                            'strategies': []
                        }
                    
                    product_ensemble_scores[product_id]['scores'].append(result.fusion_score)
                    product_ensemble_scores[product_id]['strategies'].append(result.strategy_used)
            
            # Calculate ensemble scores (average of strategies)
            fusion_results = []
            for product_id, ensemble_data in product_ensemble_scores.items():
                data = ensemble_data['data']
                scores = ensemble_data['scores']
                strategies = ensemble_data['strategies']
                
                # Ensemble score (average with variance penalty)
                ensemble_score = np.mean(scores)
                score_variance = np.var(scores)
                variance_penalty = score_variance * 0.1  # Small penalty for high variance
                final_score = ensemble_score - variance_penalty
                
                # Higher confidence for consensus among strategies
                consensus_bonus = 0.1 if len(set(strategies)) == 1 else 0
                confidence = min(0.9, final_score + consensus_bonus)
                
                fusion_result = FusionResult(
                    product_id=product_id,
                    name=data.name,
                    category=data.category,
                    brand=data.brand,
                    price=data.price,
                    rating=data.rating,
                    image_url=data.image_url,
                    text_score=data.text_score,
                    image_score=data.image_score,
                    fusion_score=final_score,
                    strategy_used=FusionStrategy.ENSEMBLE_FUSION.value,
                    confidence=confidence,
                    explanation={
                        'method': 'ensemble_fusion',
                        'strategies_used': strategies,
                        'individual_scores': scores,
                        'ensemble_score': ensemble_score,
                        'variance_penalty': variance_penalty,
                        'consensus_bonus': consensus_bonus
                    }
                )
                fusion_results.append(fusion_result)
            
            # Sort by fusion score
            fusion_results.sort(key=lambda x: x.fusion_score, reverse=True)
            return fusion_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in ensemble fusion: {e}")
            return []
    
    def _analyze_result_quality(self, results: List[Dict[str, Any]]) -> float:
        """Analyze quality of search results"""
        try:
            if not results:
                return 0.0
            
            # Quality metrics
            avg_score = np.mean([r['score'] for r in results])
            score_variance = np.var([r['score'] for r in results])
            top_score = max([r['score'] for r in results])
            
            # Quality score (higher average, lower variance, higher top score)
            quality = (avg_score * 0.4 + (1 - score_variance) * 0.3 + top_score * 0.3)
            
            return min(1.0, quality)
            
        except Exception as e:
            logger.error(f"Error analyzing result quality: {e}")
            return 0.0
    
    def _calculate_confidence(self, text_score: float, image_score: float) -> float:
        """Calculate confidence in fusion result"""
        try:
            # Higher confidence when both modalities agree
            if text_score > 0 and image_score > 0:
                # Both modalities present
                agreement = 1 - abs(text_score - image_score)
                confidence = (text_score + image_score) / 2 * (0.7 + 0.3 * agreement)
            elif text_score > 0:
                # Text only
                confidence = text_score * 0.8
            elif image_score > 0:
                # Image only
                confidence = image_score * 0.8
            else:
                # Neither modality
                confidence = 0.0
            
            return min(1.0, confidence)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.0
    
    def _calculate_cross_modal_score(self, text_result: Dict[str, Any], 
                                   visual_result: Dict[str, Any]) -> float:
        """Calculate cross-modal score"""
        try:
            # Base scores
            text_score = text_result['score']
            visual_score = visual_result['score']
            
            # Cross-modal boost when both are high
            if text_score > 0.7 and visual_score > 0.7:
                boost = 0.2
            elif text_score > 0.5 and visual_score > 0.5:
                boost = 0.1
            else:
                boost = 0.0
            
            # Weighted combination with boost
            cross_modal_score = (text_score * 0.5 + visual_score * 0.5) + boost
            
            return min(1.0, cross_modal_score)
            
        except Exception as e:
            logger.error(f"Error calculating cross-modal score: {e}")
            return 0.0
    
    def _calculate_cross_modal_confidence(self, text_result: Dict[str, Any], 
                                        visual_result: Dict[str, Any]) -> float:
        """Calculate confidence in cross-modal match"""
        try:
            text_score = text_result['score']
            visual_score = visual_result['score']
            
            # Higher confidence when scores are similar and high
            score_diff = abs(text_score - visual_score)
            avg_score = (text_score + visual_score) / 2
            
            confidence = avg_score * (1 - score_diff) * 1.2  # Boost for cross-modal
            
            return min(1.0, confidence)
            
        except Exception as e:
            logger.error(f"Error calculating cross-modal confidence: {e}")
            return 0.0
    
    def _find_best_visual_match(self, text_result: Dict[str, Any], 
                              image_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find best visual match for text result"""
        try:
            product_id = text_result['product_id']
            
            # Look for exact match
            for result in image_results:
                if result['product_id'] == product_id:
                    return result
            
            # Look for similar products (same category/brand)
            best_match = None
            best_score = 0.0
            
            for result in image_results:
                similarity = self._calculate_product_similarity(text_result, result)
                if similarity > best_score:
                    best_score = similarity
                    best_match = result
            
            return best_match if best_score > 0.5 else None
            
        except Exception as e:
            logger.error(f"Error finding best visual match: {e}")
            return None
    
    def _calculate_product_similarity(self, text_result: Dict[str, Any], 
                                    image_result: Dict[str, Any]) -> float:
        """Calculate similarity between text and image result products"""
        try:
            similarity = 0.0
            
            # Category match
            if text_result['category'] == image_result['category']:
                similarity += 0.4
            
            # Brand match
            if text_result['brand'] == image_result['brand']:
                similarity += 0.3
            
            # Price range similarity
            price_diff = abs(text_result['price'] - image_result['price'])
            price_similarity = max(0, 1 - price_diff / max(text_result['price'], image_result['price']))
            similarity += price_similarity * 0.2
            
            # Rating similarity
            rating_diff = abs(text_result['rating'] - image_result['rating'])
            rating_similarity = max(0, 1 - rating_diff / 5.0)
            similarity += rating_similarity * 0.1
            
            return similarity
            
        except Exception as e:
            logger.error(f"Error calculating product similarity: {e}")
            return 0.0
    
    def _extract_text_features(self, result: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from text search result"""
        try:
            features = {}
            
            # Score features
            features['text_score'] = result['score']
            features['semantic_score'] = result.get('semantic_score', 0)
            
            # Category features
            features['is_clothing'] = 1.0 if result['category'] == 'Clothing' else 0.0
            features['is_electronics'] = 1.0 if result['category'] == 'Electronics' else 0.0
            features['is_accessories'] = 1.0 if result['category'] == 'Accessories' else 0.0
            
            # Price features
            features['price_normalized'] = min(1.0, result['price'] / 200.0)
            features['is_cheap'] = 1.0 if result['price'] < 50 else 0.0
            features['is_expensive'] = 1.0 if result['price'] > 100 else 0.0
            
            # Rating features
            features['rating_normalized'] = result['rating'] / 5.0
            features['is_high_rated'] = 1.0 if result['rating'] > 4.0 else 0.0
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting text features: {e}")
            return {}
    
    def _extract_image_features(self, result: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from image search result"""
        try:
            features = {}
            
            # Score features
            features['image_score'] = result['score']
            features['similarity_score'] = result.get('similarity_score', 0)
            
            # Visual features (simplified)
            features['visual_confidence'] = min(1.0, result['score'] * 1.2)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting image features: {e}")
            return {}
    
    def _simulate_neural_fusion(self, text_score: float, image_score: float, 
                              features: Dict[str, float]) -> float:
        """Simulate neural network fusion"""
        try:
            # Simulated neural network computation
            # In production, this would be a trained neural network
            
            # Feature weights (simulated learned weights)
            weights = {
                'text_score': 0.3,
                'image_score': 0.4,
                'semantic_score': 0.1,
                'visual_confidence': 0.1,
                'rating_normalized': 0.05,
                'price_normalized': 0.05
            }
            
            # Weighted sum
            neural_input = text_score * weights['text_score'] + image_score * weights['image_score']
            
            # Add feature contributions
            for feature, value in features.items():
                if feature in weights:
                    neural_input += value * weights[feature]
            
            # Apply activation function (simulated ReLU with normalization)
            neural_output = max(0, neural_input)
            neural_output = min(1.0, neural_output)  # Normalize to [0,1]
            
            return neural_output
            
        except Exception as e:
            logger.error(f"Error in neural fusion simulation: {e}")
            return (text_score + image_score) / 2
    
    def _update_fusion_stats(self, strategy: FusionStrategy, results: List[FusionResult], fusion_time: float):
        """Update fusion statistics"""
        try:
            self.fusion_stats['total_fusions'] += 1
            self.fusion_stats['strategy_usage'][strategy.value] += 1
            
            if results:
                avg_confidence = np.mean([r.confidence for r in results])
                self.fusion_stats['avg_confidence'] = (
                    (self.fusion_stats['avg_confidence'] * (self.fusion_stats['total_fusions'] - 1) + avg_confidence) /
                    self.fusion_stats['total_fusions']
                )
            
            self.fusion_stats['avg_fusion_time'] = (
                (self.fusion_stats['avg_fusion_time'] * (self.fusion_stats['total_fusions'] - 1) + fusion_time) /
                self.fusion_stats['total_fusions']
            )
            
        except Exception as e:
            logger.error(f"Error updating fusion stats: {e}")
    
    def get_fusion_analytics(self) -> Dict[str, Any]:
        """Get fusion analytics and performance metrics"""
        try:
            return {
                'total_fusions': self.fusion_stats['total_fusions'],
                'strategy_distribution': self.fusion_stats['strategy_usage'],
                'average_confidence': round(self.fusion_stats['avg_confidence'], 3),
                'average_fusion_time_ms': round(self.fusion_stats['avg_fusion_time'] * 1000, 2),
                'most_used_strategy': max(self.fusion_stats['strategy_usage'], 
                                        key=self.fusion_stats['strategy_usage'].get),
                'performance_metrics': {
                    'confidence_distribution': 'high' if self.fusion_stats['avg_confidence'] > 0.7 else 'medium',
                    'speed_rating': 'fast' if self.fusion_stats['avg_fusion_time'] < 0.5 else 'medium'
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting fusion analytics: {e}")
            return {}

# Global instance
multimodal_fusion_service = MultimodalFusionService()

def get_multimodal_fusion_service() -> MultimodalFusionService:
    """Get global multimodal fusion service instance"""
    return multimodal_fusion_service
