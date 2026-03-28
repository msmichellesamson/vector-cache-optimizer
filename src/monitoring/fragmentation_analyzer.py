import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import redis
import statistics

logger = logging.getLogger(__name__)

@dataclass
class FragmentationStats:
    """Memory fragmentation analysis results"""
    fragmentation_ratio: float
    used_memory: int
    used_memory_rss: int
    mem_fragmentation_ratio: float
    total_system_memory: int
    severity: str
    recommendations: List[str]
    timestamp: datetime

class FragmentationAnalyzer:
    """Analyzes Redis memory fragmentation patterns"""
    
    def __init__(self, redis_client: redis.Redis, threshold: float = 1.5):
        self.redis = redis_client
        self.threshold = threshold
        self.history: List[FragmentationStats] = []
    
    def analyze_fragmentation(self) -> FragmentationStats:
        """Analyze current memory fragmentation"""
        try:
            info = self.redis.info('memory')
            
            used_memory = info.get('used_memory', 0)
            used_memory_rss = info.get('used_memory_rss', 0)
            total_system_memory = info.get('total_system_memory', 0)
            
            # Calculate fragmentation ratio
            if used_memory > 0:
                fragmentation_ratio = used_memory_rss / used_memory
            else:
                fragmentation_ratio = 1.0
            
            mem_fragmentation_ratio = info.get('mem_fragmentation_ratio', 1.0)
            
            # Determine severity
            if fragmentation_ratio > 2.0:
                severity = 'critical'
            elif fragmentation_ratio > self.threshold:
                severity = 'warning'
            else:
                severity = 'normal'
            
            recommendations = self._generate_recommendations(
                fragmentation_ratio, used_memory, total_system_memory
            )
            
            stats = FragmentationStats(
                fragmentation_ratio=fragmentation_ratio,
                used_memory=used_memory,
                used_memory_rss=used_memory_rss,
                mem_fragmentation_ratio=mem_fragmentation_ratio,
                total_system_memory=total_system_memory,
                severity=severity,
                recommendations=recommendations,
                timestamp=datetime.utcnow()
            )
            
            self.history.append(stats)
            # Keep only last 100 measurements
            if len(self.history) > 100:
                self.history.pop(0)
            
            return stats
            
        except Exception as e:
            logger.error(f"Fragmentation analysis failed: {e}")
            raise
    
    def _generate_recommendations(self, frag_ratio: float, used_mem: int, total_mem: int) -> List[str]:
        """Generate optimization recommendations based on fragmentation"""
        recommendations = []
        
        if frag_ratio > 2.0:
            recommendations.append("CRITICAL: Consider Redis restart or MEMORY PURGE")
            recommendations.append("Review large key patterns and TTL policies")
        
        if frag_ratio > 1.5:
            recommendations.append("Enable Redis active defragmentation")
            recommendations.append("Consider shorter TTLs for large objects")
        
        memory_usage_pct = (used_mem / total_mem) * 100 if total_mem > 0 else 0
        if memory_usage_pct > 80:
            recommendations.append("Memory usage high - consider scaling or eviction")
        
        return recommendations
    
    def get_trend_analysis(self, hours: int = 24) -> Optional[Dict]:
        """Analyze fragmentation trends over time"""
        if len(self.history) < 2:
            return None
        
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        recent_stats = [s for s in self.history if s.timestamp > cutoff]
        
        if not recent_stats:
            return None
        
        ratios = [s.fragmentation_ratio for s in recent_stats]
        
        return {
            'avg_fragmentation': statistics.mean(ratios),
            'max_fragmentation': max(ratios),
            'min_fragmentation': min(ratios),
            'trend_direction': 'increasing' if ratios[-1] > ratios[0] else 'decreasing',
            'samples': len(ratios)
        }