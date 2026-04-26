"""
Advanced ranking service with contextual understanding
"""
import os
import logging
import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime, timedelta

from .multimodal_fusion import FusionResult, get_multimodal_fusion_service
from .cross_modal_retrieval import get_cross_modal_retrieval_service

logger = logging.getLogger(__name__)

class RankingContext(Enum):
    """Ranking context types"""
    GENERAL_SEARCH = "general_search"
    CATEGORY_SPECIFIC = "category_specific"
    BRAND_PREFERENCE = "brand_preference"
    PRICE_SENSITIVE = "price_sensitive"
    QUALITY_FOCUSED = "quality_focused"
    TRENDING_ITEMS = "trending_items"
    SEASONAL_SEARCH = "seasonal_search"

@dataclass
class RankingFactors:
    """Factors for advanced ranking"""
    relevance_score: float
    popularity_score: float
    quality_score: float
    price_score: float
    category_match: float
    brand_preference: float
    seasonal_boost: float
    trending_boost: float
    user_preference: float
    contextual_boost: float

@dataclass
class RankedResult:
    """Ranked search result with detailed scoring"""
    product_id: int
    name: str
    category: str
    brand: str
    price: float
    rating: float
    image_url: str
    base_score: float
    final_score: float
    ranking_factors: RankingFactors
    context_used: str
    explanation: Dict[str, Any]
    rank_position: int

class AdvancedRankingService:
    """Service for advanced ranking with contextual understanding"""
    
    def __init__(self):
        self.multimodal_fusion_service = get_multimodal_fusion_service()
        self.cross_modal_service = get_cross_modal_retrieval_service()
        
        # Ranking weights (configurable)
        self.ranking_weights = {
            'relevance': 0.4,
            'popularity': 0.2,
            'quality': 0.15,
            'price': 0.1,
            'context': 0.15
        }
        
        # Contextual factors
        self.seasonal_factors = self._initialize_seasonal_factors()
        self.trending_data = {}
        self.user_preferences = {}
        
        # Performance tracking
        self.ranking_stats = {
            'total_rankings': 0,
            'context_usage': {context.value: 0 for context in RankingContext},
            'avg_ranking_time': 0.0,
            'context_effectiveness': {}
        }
    
    def rank_search_results(self, 
                          fusion_results: List[FusionResult],
                          context: RankingContext = RankingContext.GENERAL_SEARCH,
                          user_context: Optional[Dict[str, Any]] = None,
                          top_k: int = 10) -> List[RankedResult]:
        """Rank search results with contextual understanding"""
        try:
            start_time = time.time()
            
            # Calculate ranking factors for each result
            ranked_results = []
            for result in fusion_results:
                ranking_factors = self._calculate_ranking_factors(
                    result, context, user_context
                )
                
                # Calculate final score
                final_score = self._calculate_final_score(ranking_factors)
                
                ranked_result = RankedResult(
                    product_id=result.product_id,
                    name=result.name,
                    category=result.category,
                    brand=result.brand,
                    price=result.price,
                    rating=result.rating,
                    image_url=result.image_url,
                    base_score=result.fusion_score,
                    final_score=final_score,
                    ranking_factors=ranking_factors,
                    context_used=context.value,
                    explanation=self._generate_ranking_explanation(ranking_factors, context),
                    rank_position=0  # Will be set after sorting
                )
                ranked_results.append(ranked_result)
            
            # Apply contextual reordering
            ranked_results = self._apply_contextual_reordering(ranked_results, context, user_context)
            
            # Sort by final score and assign ranks
            ranked_results.sort(key=lambda x: x.final_score, reverse=True)
            for i, result in enumerate(ranked_results):
                result.rank_position = i + 1
            
            # Limit to top_k
            ranked_results = ranked_results[:top_k]
            
            # Update stats
            ranking_time = time.time() - start_time
            self._update_ranking_stats(context, ranked_results, ranking_time)
            
            return ranked_results
            
        except Exception as e:
            logger.error(f"Error in advanced ranking: {e}")
            return []
    
    def _calculate_ranking_factors(self, result: FusionResult, 
                                 context: RankingContext,
                                 user_context: Optional[Dict[str, Any]]) -> RankingFactors:
        """Calculate ranking factors for a result"""
        try:
            # Base relevance score
            relevance_score = result.fusion_score
            
            # Popularity score (based on rating and number of reviews)
            popularity_score = self._calculate_popularity_score(result.rating)
            
            # Quality score (rating + confidence)
            quality_score = self._calculate_quality_score(result.rating, result.confidence)
            
            # Price score (context-dependent)
            price_score = self._calculate_price_score(result.price, context, user_context)
            
            # Category match
            category_match = self._calculate_category_match(result.category, user_context)
            
            # Brand preference
            brand_preference = self._calculate_brand_preference(result.brand, user_context)
            
            # Seasonal boost
            seasonal_boost = self._calculate_seasonal_boost(result.category, result.name)
            
            # Trending boost
            trending_boost = self._calculate_trending_boost(result.product_id)
            
            # User preference
            user_preference = self._calculate_user_preference(result, user_context)
            
            # Contextual boost
            contextual_boost = self._calculate_contextual_boost(result, context, user_context)
            
            return RankingFactors(
                relevance_score=relevance_score,
                popularity_score=popularity_score,
                quality_score=quality_score,
                price_score=price_score,
                category_match=category_match,
                brand_preference=brand_preference,
                seasonal_boost=seasonal_boost,
                trending_boost=trending_boost,
                user_preference=user_preference,
                contextual_boost=contextual_boost
            )
            
        except Exception as e:
            logger.error(f"Error calculating ranking factors: {e}")
            return RankingFactors(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    def _calculate_final_score(self, factors: RankingFactors) -> float:
        """Calculate final ranking score from factors"""
        try:
            # Weighted combination of factors
            final_score = (
                factors.relevance_score * self.ranking_weights['relevance'] +
                factors.popularity_score * self.ranking_weights['popularity'] +
                factors.quality_score * self.ranking_weights['quality'] +
                factors.price_score * self.ranking_weights['price'] +
                factors.contextual_boost * self.ranking_weights['context']
            )
            
            # Add additional boosts
            final_score += (
                factors.category_match * 0.05 +
                factors.brand_preference * 0.03 +
                factors.seasonal_boost * 0.02 +
                factors.trending_boost * 0.03 +
                factors.user_preference * 0.02
            )
            
            return min(1.0, final_score)
            
        except Exception as e:
            logger.error(f"Error calculating final score: {e}")
            return 0.0
    
    def _calculate_popularity_score(self, rating: float) -> float:
        """Calculate popularity score from rating"""
        try:
            # Normalize rating to [0,1]
            return rating / 5.0
            
        except Exception as e:
            logger.error(f"Error calculating popularity score: {e}")
            return 0.0
    
    def _calculate_quality_score(self, rating: float, confidence: float) -> float:
        """Calculate quality score from rating and confidence"""
        try:
            # Combine rating and confidence
            return (rating / 5.0) * confidence
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 0.0
    
    def _calculate_price_score(self, price: float, context: RankingContext, 
                             user_context: Optional[Dict[str, Any]]) -> float:
        """Calculate price score based on context"""
        try:
            if context == RankingContext.PRICE_SENSITIVE:
                # For price-sensitive users, lower prices get higher scores
                return max(0, 1 - (price / 200.0))  # Normalize to [0,1], invert
            elif context == RankingContext.QUALITY_FOCUSED:
                # For quality-focused users, moderate-high prices get higher scores
                if price < 20:
                    return 0.3  # Too cheap might indicate low quality
                elif price < 100:
                    return 0.8  # Sweet spot
                else:
                    return 0.6  # Expensive but potentially high quality
            else:
                # General case: moderate prices preferred
                if price < 20:
                    return 0.4
                elif price < 100:
                    return 0.8
                else:
                    return 0.5
            
        except Exception as e:
            logger.error(f"Error calculating price score: {e}")
            return 0.5
    
    def _calculate_category_match(self, category: str, user_context: Optional[Dict[str, Any]]) -> float:
        """Calculate category match score"""
        try:
            if not user_context:
                return 0.5  # Default score
            
            preferred_categories = user_context.get('preferred_categories', [])
            if not preferred_categories:
                return 0.5
            
            # Exact match gets highest score
            if category in preferred_categories:
                return 1.0
            
            # Partial match gets medium score
            category_lower = category.lower()
            for preferred in preferred_categories:
                if preferred.lower() in category_lower or category_lower in preferred.lower():
                    return 0.7
            
            return 0.3  # No match
            
        except Exception as e:
            logger.error(f"Error calculating category match: {e}")
            return 0.5
    
    def _calculate_brand_preference(self, brand: str, user_context: Optional[Dict[str, Any]]) -> float:
        """Calculate brand preference score"""
        try:
            if not user_context or not brand:
                return 0.5
            
            preferred_brands = user_context.get('preferred_brands', [])
            if not preferred_brands:
                return 0.5
            
            if brand in preferred_brands:
                return 1.0
            
            return 0.3  # No preference match
            
        except Exception as e:
            logger.error(f"Error calculating brand preference: {e}")
            return 0.5
    
    def _calculate_seasonal_boost(self, category: str, name: str) -> float:
        """Calculate seasonal boost based on current season"""
        try:
            current_month = datetime.now().month
            
            # Determine current season
            if current_month in [12, 1, 2]:  # Winter
                current_season = 'winter'
                seasonal_items = ['jacket', 'coat', 'sweater', 'boots', 'gloves', 'scarf']
            elif current_month in [3, 4, 5]:  # Spring
                current_season = 'spring'
                seasonal_items = ['dress', 'light jacket', 'sneakers', 'sunglasses', 't-shirt']
            elif current_month in [6, 7, 8]:  # Summer
                current_season = 'summer'
                seasonal_items = ['shorts', 'sandals', 'swimwear', 'sunscreen', 'hat', 'dress']
            else:  # Fall
                current_season = 'fall'
                seasonal_items = ['jeans', 'boots', 'sweater', 'jacket', 'scarf']
            
            # Check if item is seasonal
            name_lower = name.lower()
            category_lower = category.lower()
            
            for item in seasonal_items:
                if item in name_lower or item in category_lower:
                    return 0.2  # Seasonal boost
            
            return 0.0  # No seasonal boost
            
        except Exception as e:
            logger.error(f"Error calculating seasonal boost: {e}")
            return 0.0
    
    def _calculate_trending_boost(self, product_id: int) -> float:
        """Calculate trending boost for product"""
        try:
            # This would typically use real trending data
            # For now, use a simple hash-based simulation
            trending_score = (product_id % 100) / 100.0
            
            # Only apply boost to top 20% trending items
            if trending_score > 0.8:
                return 0.15
            elif trending_score > 0.6:
                return 0.10
            else:
                return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating trending boost: {e}")
            return 0.0
    
    def _calculate_user_preference(self, result: FusionResult, user_context: Optional[Dict[str, Any]]) -> float:
        """Calculate user preference score"""
        try:
            if not user_context:
                return 0.5
            
            # Check if user has interacted with similar items
            interaction_history = user_context.get('interaction_history', [])
            if not interaction_history:
                return 0.5
            
            # Simple similarity check based on category and brand
            user_categories = [item.get('category') for item in interaction_history]
            user_brands = [item.get('brand') for item in interaction_history]
            
            category_match = 1.0 if result.category in user_categories else 0.0
            brand_match = 1.0 if result.brand in user_brands else 0.0
            
            return (category_match * 0.7 + brand_match * 0.3)
            
        except Exception as e:
            logger.error(f"Error calculating user preference: {e}")
            return 0.5
    
    def _calculate_contextual_boost(self, result: FusionResult, context: RankingContext, 
                                  user_context: Optional[Dict[str, Any]]) -> float:
        """Calculate contextual boost based on ranking context"""
        try:
            if context == RankingContext.CATEGORY_SPECIFIC:
                # Boost items matching the specific category
                target_category = user_context.get('target_category') if user_context else None
                if target_category and result.category == target_category:
                    return 0.2
                return 0.0
            
            elif context == RankingContext.BRAND_PREFERENCE:
                # Boost preferred brands
                if user_context and result.brand in user_context.get('preferred_brands', []):
                    return 0.15
                return 0.0
            
            elif context == RankingContext.TRENDING_ITEMS:
                # Boost trending items
                return self._calculate_trending_boost(result.product_id)
            
            elif context == RankingContext.SEASONAL_SEARCH:
                # Boost seasonal items
                return self._calculate_seasonal_boost(result.category, result.name)
            
            else:
                return 0.0  # No specific contextual boost
            
        except Exception as e:
            logger.error(f"Error calculating contextual boost: {e}")
            return 0.0
    
    def _apply_contextual_reordering(self, results: List[RankedResult], 
                                   context: RankingContext,
                                   user_context: Optional[Dict[str, Any]]) -> List[RankedResult]:
        """Apply contextual reordering to results"""
        try:
            if context == RankingContext.QUALITY_FOCUSED:
                # Prioritize high-quality items
                results.sort(key=lambda x: x.ranking_factors.quality_score, reverse=True)
            
            elif context == RankingContext.PRICE_SENSITIVE:
                # Prioritize affordable items
                results.sort(key=lambda x: x.price)
            
            elif context == RankingContext.TRENDING_ITEMS:
                # Prioritize trending items
                results.sort(key=lambda x: x.ranking_factors.trending_boost, reverse=True)
            
            # For other contexts, keep the original ordering based on final score
            return results
            
        except Exception as e:
            logger.error(f"Error applying contextual reordering: {e}")
            return results
    
    def _generate_ranking_explanation(self, factors: RankingFactors, context: RankingContext) -> Dict[str, Any]:
        """Generate explanation for ranking decision"""
        try:
            explanation = {
                'context': context.value,
                'primary_factors': [],
                'boosts_applied': [],
                'final_score_components': {}
            }
            
            # Identify primary factors
            if factors.relevance_score > 0.7:
                explanation['primary_factors'].append('high_relevance')
            if factors.quality_score > 0.7:
                explanation['primary_factors'].append('high_quality')
            if factors.popularity_score > 0.7:
                explanation['primary_factors'].append('popular')
            
            # Identify boosts
            if factors.seasonal_boost > 0:
                explanation['boosts_applied'].append('seasonal_boost')
            if factors.trending_boost > 0:
                explanation['boosts_applied'].append('trending_boost')
            if factors.user_preference > 0.7:
                explanation['boosts_applied'].append('user_preference')
            
            # Score components
            explanation['final_score_components'] = {
                'relevance': round(factors.relevance_score * self.ranking_weights['relevance'], 3),
                'popularity': round(factors.popularity_score * self.ranking_weights['popularity'], 3),
                'quality': round(factors.quality_score * self.ranking_weights['quality'], 3),
                'price': round(factors.price_score * self.ranking_weights['price'], 3),
                'context': round(factors.contextual_boost * self.ranking_weights['context'], 3)
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating ranking explanation: {e}")
            return {'error': str(e)}
    
    def _initialize_seasonal_factors(self) -> Dict[str, float]:
        """Initialize seasonal factors"""
        return {
            'winter': 1.2,
            'spring': 1.0,
            'summer': 1.1,
            'fall': 1.0
        }
    
    def _update_ranking_stats(self, context: RankingContext, results: List[RankedResult], ranking_time: float):
        """Update ranking statistics"""
        try:
            self.ranking_stats['total_rankings'] += 1
            self.ranking_stats['context_usage'][context.value] += 1
            
            # Update average ranking time
            self.ranking_stats['avg_ranking_time'] = (
                (self.ranking_stats['avg_ranking_time'] * (self.ranking_stats['total_rankings'] - 1) + ranking_time) /
                self.ranking_stats['total_rankings']
            )
            
            # Update context effectiveness
            if results:
                avg_score = np.mean([r.final_score for r in results])
                if context.value not in self.ranking_stats['context_effectiveness']:
                    self.ranking_stats['context_effectiveness'][context.value] = []
                self.ranking_stats['context_effectiveness'][context.value].append(avg_score)
            
        except Exception as e:
            logger.error(f"Error updating ranking stats: {e}")
    
    def get_ranking_analytics(self) -> Dict[str, Any]:
        """Get ranking analytics and performance metrics"""
        try:
            analytics = {
                'total_rankings': self.ranking_stats['total_rankings'],
                'context_distribution': self.ranking_stats['context_usage'],
                'average_ranking_time_ms': round(self.ranking_stats['avg_ranking_time'] * 1000, 2),
                'most_used_context': max(self.ranking_stats['context_usage'], 
                                        key=self.ranking_stats['context_usage'].get),
                'ranking_weights': self.ranking_weights,
                'context_effectiveness': {}
            }
            
            # Calculate context effectiveness
            for context, scores in self.ranking_stats['context_effectiveness'].items():
                if scores:
                    analytics['context_effectiveness'][context] = {
                        'average_score': round(np.mean(scores), 3),
                        'score_variance': round(np.var(scores), 3),
                        'total_usages': len(scores)
                    }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting ranking analytics: {e}")
            return {}
    
    def update_ranking_weights(self, new_weights: Dict[str, float]):
        """Update ranking weights"""
        try:
            # Validate weights sum to 1
            total_weight = sum(new_weights.values())
            if abs(total_weight - 1.0) > 0.01:
                logger.warning(f"Weights sum to {total_weight}, normalizing...")
                # Normalize weights
                for key in new_weights:
                    new_weights[key] = new_weights[key] / total_weight
            
            self.ranking_weights.update(new_weights)
            logger.info(f"Updated ranking weights: {self.ranking_weights}")
            
        except Exception as e:
            logger.error(f"Error updating ranking weights: {e}")

# Global instance
advanced_ranking_service = AdvancedRankingService()

def get_advanced_ranking_service() -> AdvancedRankingService:
    """Get global advanced ranking service instance"""
    return advanced_ranking_service
