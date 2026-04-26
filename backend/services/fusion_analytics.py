"""
Fusion analytics and performance metrics service
"""
import os
import logging
import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import json

from .multimodal_fusion import get_multimodal_fusion_service, FusionStrategy
from .cross_modal_retrieval import get_cross_modal_retrieval_service
from .advanced_ranking import get_advanced_ranking_service, RankingContext
from .performance_optimizer import get_performance_optimizer

logger = logging.getLogger(__name__)

@dataclass
class FusionMetric:
    """Individual fusion metric"""
    timestamp: datetime
    strategy: str
    execution_time_ms: float
    fusion_score: float
    confidence: float
    result_count: int
    text_available: bool
    image_available: bool
    cache_hit: bool

@dataclass
class CrossModalMetric:
    """Cross-modal retrieval metric"""
    timestamp: datetime
    query_type: str
    execution_time_ms: float
    cross_modal_score: float
    confidence: float
    result_count: int
    cache_hit: bool

@dataclass
class RankingMetric:
    """Advanced ranking metric"""
    timestamp: datetime
    context: str
    execution_time_ms: float
    avg_final_score: float
    result_count: int
    top_score: float
    score_variance: float

class FusionAnalyticsService:
    """Service for fusion analytics and performance metrics"""
    
    def __init__(self):
        self.multimodal_fusion_service = get_multimodal_fusion_service()
        self.cross_modal_service = get_cross_modal_retrieval_service()
        self.ranking_service = get_advanced_ranking_service()
        self.performance_optimizer = get_performance_optimizer()
        
        # Analytics storage
        self.fusion_metrics = []
        self.cross_modal_metrics = []
        self.ranking_metrics = []
        
        # Analytics configuration
        self.max_metrics_history = 10000
        self.analytics_retention_days = 30
        
        # Performance thresholds
        self.performance_thresholds = {
            'slow_fusion_ms': 200,
            'low_confidence': 0.5,
            'poor_fusion_score': 0.3,
            'high_variance': 0.2
        }
    
    def record_fusion_metric(self, strategy: str, execution_time_ms: float, 
                           fusion_score: float, confidence: float, 
                           result_count: int, text_available: bool, 
                           image_available: bool, cache_hit: bool = False):
        """Record fusion operation metric"""
        try:
            metric = FusionMetric(
                timestamp=datetime.now(),
                strategy=strategy,
                execution_time_ms=execution_time_ms,
                fusion_score=fusion_score,
                confidence=confidence,
                result_count=result_count,
                text_available=text_available,
                image_available=image_available,
                cache_hit=cache_hit
            )
            
            self.fusion_metrics.append(metric)
            self._cleanup_old_metrics()
            
            # Alert on performance issues
            self._check_fusion_performance(metric)
            
        except Exception as e:
            logger.error(f"Error recording fusion metric: {e}")
    
    def record_cross_modal_metric(self, query_type: str, execution_time_ms: float,
                                cross_modal_score: float, confidence: float,
                                result_count: int, cache_hit: bool = False):
        """Record cross-modal retrieval metric"""
        try:
            metric = CrossModalMetric(
                timestamp=datetime.now(),
                query_type=query_type,
                execution_time_ms=execution_time_ms,
                cross_modal_score=cross_modal_score,
                confidence=confidence,
                result_count=result_count,
                cache_hit=cache_hit
            )
            
            self.cross_modal_metrics.append(metric)
            self._cleanup_old_metrics()
            
            # Alert on performance issues
            self._check_cross_modal_performance(metric)
            
        except Exception as e:
            logger.error(f"Error recording cross-modal metric: {e}")
    
    def record_ranking_metric(self, context: str, execution_time_ms: float,
                            avg_final_score: float, result_count: int,
                            top_score: float, score_variance: float):
        """Record advanced ranking metric"""
        try:
            metric = RankingMetric(
                timestamp=datetime.now(),
                context=context,
                execution_time_ms=execution_time_ms,
                avg_final_score=avg_final_score,
                result_count=result_count,
                top_score=top_score,
                score_variance=score_variance
            )
            
            self.ranking_metrics.append(metric)
            self._cleanup_old_metrics()
            
            # Alert on performance issues
            self._check_ranking_performance(metric)
            
        except Exception as e:
            logger.error(f"Error recording ranking metric: {e}")
    
    def get_fusion_analytics(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive fusion analytics"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=time_range_hours)
            recent_metrics = [m for m in self.fusion_metrics if m.timestamp > cutoff_time]
            
            if not recent_metrics:
                return {'message': 'No fusion metrics available in specified time range'}
            
            analytics = {
                'time_range_hours': time_range_hours,
                'total_fusions': len(recent_metrics),
                'strategy_performance': {},
                'performance_metrics': {},
                'quality_metrics': {},
                'modality_usage': {},
                'trends': {}
            }
            
            # Strategy performance
            strategy_groups = defaultdict(list)
            for metric in recent_metrics:
                strategy_groups[metric.strategy].append(metric)
            
            for strategy, metrics in strategy_groups.items():
                execution_times = [m.execution_time_ms for m in metrics]
                fusion_scores = [m.fusion_score for m in metrics]
                confidences = [m.confidence for m in metrics]
                
                analytics['strategy_performance'][strategy] = {
                    'usage_count': len(metrics),
                    'avg_execution_time_ms': round(np.mean(execution_times), 2),
                    'avg_fusion_score': round(np.mean(fusion_scores), 3),
                    'avg_confidence': round(np.mean(confidences), 3),
                    'success_rate': round(sum(1 for m in metrics if m.confidence > 0.5) / len(metrics), 3)
                }
            
            # Performance metrics
            all_execution_times = [m.execution_time_ms for m in recent_metrics]
            analytics['performance_metrics'] = {
                'avg_execution_time_ms': round(np.mean(all_execution_times), 2),
                'median_execution_time_ms': round(np.median(all_execution_times), 2),
                'p95_execution_time_ms': round(np.percentile(all_execution_times, 95), 2),
                'slow_operations': sum(1 for m in recent_metrics if m.execution_time_ms > self.performance_thresholds['slow_fusion_ms']),
                'cache_hit_rate': round(sum(1 for m in recent_metrics if m.cache_hit) / len(recent_metrics), 3)
            }
            
            # Quality metrics
            all_fusion_scores = [m.fusion_score for m in recent_metrics]
            all_confidences = [m.confidence for m in recent_metrics]
            analytics['quality_metrics'] = {
                'avg_fusion_score': round(np.mean(all_fusion_scores), 3),
                'avg_confidence': round(np.mean(all_confidences), 3),
                'high_confidence_rate': round(sum(1 for m in recent_metrics if m.confidence > 0.7) / len(recent_metrics), 3),
                'low_confidence_rate': round(sum(1 for m in recent_metrics if m.confidence < 0.3) / len(recent_metrics), 3),
                'score_variance': round(np.var(all_fusion_scores), 4)
            }
            
            # Modality usage
            text_only = sum(1 for m in recent_metrics if m.text_available and not m.image_available)
            image_only = sum(1 for m in recent_metrics if m.image_available and not m.text_available)
            both_modalities = sum(1 for m in recent_metrics if m.text_available and m.image_available)
            
            analytics['modality_usage'] = {
                'text_only': text_only,
                'image_only': image_only,
                'both_modalities': both_modalities,
                'text_only_rate': round(text_only / len(recent_metrics), 3),
                'image_only_rate': round(image_only / len(recent_metrics), 3),
                'both_modalities_rate': round(both_modalities / len(recent_metrics), 3)
            }
            
            # Trends (hourly)
            hourly_groups = defaultdict(list)
            for metric in recent_metrics:
                hour = metric.timestamp.hour
                hourly_groups[hour].append(metric)
            
            analytics['trends']['hourly_usage'] = {
                str(hour): len(metrics) for hour, metrics in hourly_groups.items()
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting fusion analytics: {e}")
            return {'error': str(e)}
    
    def get_cross_modal_analytics(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """Get cross-modal retrieval analytics"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=time_range_hours)
            recent_metrics = [m for m in self.cross_modal_metrics if m.timestamp > cutoff_time]
            
            if not recent_metrics:
                return {'message': 'No cross-modal metrics available in specified time range'}
            
            analytics = {
                'time_range_hours': time_range_hours,
                'total_retrievals': len(recent_metrics),
                'query_type_performance': {},
                'performance_metrics': {},
                'quality_metrics': {}
            }
            
            # Query type performance
            query_type_groups = defaultdict(list)
            for metric in recent_metrics:
                query_type_groups[metric.query_type].append(metric)
            
            for query_type, metrics in query_type_groups.items():
                execution_times = [m.execution_time_ms for m in metrics]
                scores = [m.cross_modal_score for m in metrics]
                confidences = [m.confidence for m in metrics]
                
                analytics['query_type_performance'][query_type] = {
                    'usage_count': len(metrics),
                    'avg_execution_time_ms': round(np.mean(execution_times), 2),
                    'avg_cross_modal_score': round(np.mean(scores), 3),
                    'avg_confidence': round(np.mean(confidences), 3),
                    'success_rate': round(sum(1 for m in metrics if m.confidence > 0.5) / len(metrics), 3)
                }
            
            # Performance metrics
            all_execution_times = [m.execution_time_ms for m in recent_metrics]
            analytics['performance_metrics'] = {
                'avg_execution_time_ms': round(np.mean(all_execution_times), 2),
                'median_execution_time_ms': round(np.median(all_execution_times), 2),
                'cache_hit_rate': round(sum(1 for m in recent_metrics if m.cache_hit) / len(recent_metrics), 3)
            }
            
            # Quality metrics
            all_scores = [m.cross_modal_score for m in recent_metrics]
            all_confidences = [m.confidence for m in recent_metrics]
            analytics['quality_metrics'] = {
                'avg_cross_modal_score': round(np.mean(all_scores), 3),
                'avg_confidence': round(np.mean(all_confidences), 3),
                'high_quality_rate': round(sum(1 for m in recent_metrics if m.cross_modal_score > 0.7) / len(recent_metrics), 3)
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting cross-modal analytics: {e}")
            return {'error': str(e)}
    
    def get_ranking_analytics(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """Get advanced ranking analytics"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=time_range_hours)
            recent_metrics = [m for m in self.ranking_metrics if m.timestamp > cutoff_time]
            
            if not recent_metrics:
                return {'message': 'No ranking metrics available in specified time range'}
            
            analytics = {
                'time_range_hours': time_range_hours,
                'total_rankings': len(recent_metrics),
                'context_performance': {},
                'performance_metrics': {},
                'quality_metrics': {}
            }
            
            # Context performance
            context_groups = defaultdict(list)
            for metric in recent_metrics:
                context_groups[metric.context].append(metric)
            
            for context, metrics in context_groups.items():
                execution_times = [m.execution_time_ms for m in metrics]
                final_scores = [m.avg_final_score for m in metrics]
                variances = [m.score_variance for m in metrics]
                
                analytics['context_performance'][context] = {
                    'usage_count': len(metrics),
                    'avg_execution_time_ms': round(np.mean(execution_times), 2),
                    'avg_final_score': round(np.mean(final_scores), 3),
                    'avg_score_variance': round(np.mean(variances), 4),
                    'consistency_rate': round(sum(1 for m in metrics if m.score_variance < 0.1) / len(metrics), 3)
                }
            
            # Performance metrics
            all_execution_times = [m.execution_time_ms for m in recent_metrics]
            analytics['performance_metrics'] = {
                'avg_execution_time_ms': round(np.mean(all_execution_times), 2),
                'median_execution_time_ms': round(np.median(all_execution_times), 2),
                'slow_rankings': sum(1 for m in recent_metrics if m.execution_time_ms > 150)
            }
            
            # Quality metrics
            all_final_scores = [m.avg_final_score for m in recent_metrics]
            all_top_scores = [m.top_score for m in recent_metrics]
            analytics['quality_metrics'] = {
                'avg_final_score': round(np.mean(all_final_scores), 3),
                'avg_top_score': round(np.mean(all_top_scores), 3),
                'high_score_rate': round(sum(1 for m in recent_metrics if m.avg_final_score > 0.7) / len(recent_metrics), 3)
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting ranking analytics: {e}")
            return {'error': str(e)}
    
    def get_comprehensive_analytics(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive analytics across all services"""
        try:
            return {
                'fusion_analytics': self.get_fusion_analytics(time_range_hours),
                'cross_modal_analytics': self.get_cross_modal_analytics(time_range_hours),
                'ranking_analytics': self.get_ranking_analytics(time_range_hours),
                'performance_analytics': self.performance_optimizer.get_performance_stats(),
                'cache_analytics': self.performance_optimizer.get_cache_stats(),
                'service_health': self._get_service_health()
            }
            
        except Exception as e:
            logger.error(f"Error getting comprehensive analytics: {e}")
            return {'error': str(e)}
    
    def _get_service_health(self) -> Dict[str, Any]:
        """Get overall service health status"""
        try:
            health = {
                'overall_status': 'healthy',
                'services': {},
                'alerts': []
            }
            
            # Check fusion service
            fusion_stats = self.multimodal_fusion_service.get_fusion_analytics()
            health['services']['multimodal_fusion'] = {
                'status': 'healthy',
                'total_fusions': fusion_stats.get('total_fusions', 0),
                'avg_confidence': fusion_stats.get('average_confidence', 0)
            }
            
            # Check cross-modal service
            cross_modal_stats = self.cross_modal_service.get_cross_modal_analytics()
            health['services']['cross_modal_retrieval'] = {
                'status': 'healthy',
                'total_queries': cross_modal_stats.get('total_queries', 0),
                'avg_score': cross_modal_stats.get('average_cross_modal_score', 0)
            }
            
            # Check ranking service
            ranking_stats = self.ranking_service.get_ranking_analytics()
            health['services']['advanced_ranking'] = {
                'status': 'healthy',
                'total_rankings': ranking_stats.get('total_rankings', 0),
                'avg_ranking_time': ranking_stats.get('average_ranking_time_ms', 0)
            }
            
            # Check performance
            perf_stats = self.performance_optimizer.get_performance_stats()
            cache_stats = self.performance_optimizer.get_cache_stats()
            
            health['services']['performance_optimizer'] = {
                'status': 'healthy',
                'total_operations': perf_stats.get('total_operations', 0),
                'cache_hit_rate': cache_stats.get('recent_hit_rate', 0)
            }
            
            # Generate alerts
            if cache_stats.get('recent_hit_rate', 0) < 0.3:
                health['alerts'].append('Low cache hit rate detected')
            
            if perf_stats.get('slow_operations'):
                health['alerts'].append(f"{len(perf_stats['slow_operations'])} slow operations detected")
            
            if health['alerts']:
                health['overall_status'] = 'warning'
            
            return health
            
        except Exception as e:
            logger.error(f"Error getting service health: {e}")
            return {'overall_status': 'error', 'error': str(e)}
    
    def _cleanup_old_metrics(self):
        """Clean up old metrics to prevent memory issues"""
        try:
            cutoff_time = datetime.now() - timedelta(days=self.analytics_retention_days)
            
            # Clean fusion metrics
            self.fusion_metrics = [m for m in self.fusion_metrics if m.timestamp > cutoff_time]
            
            # Clean cross-modal metrics
            self.cross_modal_metrics = [m for m in self.cross_modal_metrics if m.timestamp > cutoff_time]
            
            # Clean ranking metrics
            self.ranking_metrics = [m for m in self.ranking_metrics if m.timestamp > cutoff_time]
            
            # Limit total number of metrics
            if len(self.fusion_metrics) > self.max_metrics_history:
                self.fusion_metrics = self.fusion_metrics[-self.max_metrics_history:]
            
            if len(self.cross_modal_metrics) > self.max_metrics_history:
                self.cross_modal_metrics = self.cross_modal_metrics[-self.max_metrics_history:]
            
            if len(self.ranking_metrics) > self.max_metrics_history:
                self.ranking_metrics = self.ranking_metrics[-self.max_metrics_history:]
                
        except Exception as e:
            logger.error(f"Error cleaning up old metrics: {e}")
    
    def _check_fusion_performance(self, metric: FusionMetric):
        """Check fusion performance and generate alerts"""
        try:
            if metric.execution_time_ms > self.performance_thresholds['slow_fusion_ms']:
                logger.warning(f"Slow fusion operation: {metric.strategy} took {metric.execution_time_ms:.2f}ms")
            
            if metric.confidence < self.performance_thresholds['low_confidence']:
                logger.warning(f"Low confidence fusion: {metric.strategy} confidence {metric.confidence:.3f}")
            
            if metric.fusion_score < self.performance_thresholds['poor_fusion_score']:
                logger.warning(f"Poor fusion score: {metric.strategy} score {metric.fusion_score:.3f}")
                
        except Exception as e:
            logger.error(f"Error checking fusion performance: {e}")
    
    def _check_cross_modal_performance(self, metric: CrossModalMetric):
        """Check cross-modal performance and generate alerts"""
        try:
            if metric.execution_time_ms > 300:  # 300ms threshold for cross-modal
                logger.warning(f"Slow cross-modal operation: {metric.query_type} took {metric.execution_time_ms:.2f}ms")
            
            if metric.confidence < self.performance_thresholds['low_confidence']:
                logger.warning(f"Low confidence cross-modal: {metric.query_type} confidence {metric.confidence:.3f}")
            
            if metric.cross_modal_score < self.performance_thresholds['poor_fusion_score']:
                logger.warning(f"Poor cross-modal score: {metric.query_type} score {metric.cross_modal_score:.3f}")
                
        except Exception as e:
            logger.error(f"Error checking cross-modal performance: {e}")
    
    def _check_ranking_performance(self, metric: RankingMetric):
        """Check ranking performance and generate alerts"""
        try:
            if metric.execution_time_ms > 150:  # 150ms threshold for ranking
                logger.warning(f"Slow ranking operation: {metric.context} took {metric.execution_time_ms:.2f}ms")
            
            if metric.score_variance > self.performance_thresholds['high_variance']:
                logger.warning(f"High score variance in ranking: {metric.context} variance {metric.score_variance:.3f}")
                
        except Exception as e:
            logger.error(f"Error checking ranking performance: {e}")
    
    def export_analytics(self, time_range_hours: int = 24, format: str = 'json') -> str:
        """Export analytics data"""
        try:
            analytics = self.get_comprehensive_analytics(time_range_hours)
            
            if format.lower() == 'json':
                return json.dumps(analytics, indent=2, default=str)
            elif format.lower() == 'csv':
                # Convert to CSV format (simplified)
                csv_data = "Metric Type,Timestamp,Value\n"
                # Add CSV conversion logic here
                return csv_data
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            logger.error(f"Error exporting analytics: {e}")
            return str(e)

# Global instance
fusion_analytics_service = FusionAnalyticsService()

def get_fusion_analytics_service() -> FusionAnalyticsService:
    """Get global fusion analytics service instance"""
    return fusion_analytics_service
