from typing import Dict, List, Tuple
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

class EvictionPolicy(Enum):
    LRU = "lru"
    LFU = "lfu"
    TTL_BASED = "ttl_based"
    ML_PREDICTED = "ml_predicted"
    ADAPTIVE = "adaptive"

@dataclass
class MemoryThreshold:
    low: float = 0.7  # 70%
    medium: float = 0.85  # 85%
    high: float = 0.95  # 95%

class EvictionSelector:
    """Selects optimal eviction policy based on memory pressure and access patterns"""
    
    def __init__(self, thresholds: MemoryThreshold = None):
        self.thresholds = thresholds or MemoryThreshold()
        self._policy_performance: Dict[EvictionPolicy, float] = {
            policy: 0.0 for policy in EvictionPolicy
        }
        self._current_policy = EvictionPolicy.LRU
    
    def select_policy(self, 
                     memory_usage: float, 
                     hit_rate: float, 
                     access_pattern_variance: float) -> EvictionPolicy:
        """Select best eviction policy based on current conditions"""
        
        # Critical memory - use fastest eviction
        if memory_usage >= self.thresholds.high:
            return EvictionPolicy.LRU
        
        # High memory - balance speed and accuracy
        if memory_usage >= self.thresholds.medium:
            if access_pattern_variance > 0.8:  # High variance = unpredictable
                return EvictionPolicy.LFU
            return EvictionPolicy.TTL_BASED
        
        # Normal memory - optimize for hit rate
        if hit_rate < 0.6:  # Low hit rate
            return EvictionPolicy.ML_PREDICTED if self._ml_available() else EvictionPolicy.ADAPTIVE
        
        return self._current_policy
    
    def update_performance(self, policy: EvictionPolicy, hit_rate: float):
        """Update policy performance metrics"""
        # Exponential moving average
        alpha = 0.1
        current = self._policy_performance[policy]
        self._policy_performance[policy] = alpha * hit_rate + (1 - alpha) * current
        
        logger.debug(f"Updated {policy.value} performance: {self._policy_performance[policy]:.3f}")
    
    def get_policy_rankings(self) -> List[Tuple[EvictionPolicy, float]]:
        """Get policies ranked by performance"""
        return sorted(self._policy_performance.items(), key=lambda x: x[1], reverse=True)
    
    def _ml_available(self) -> bool:
        """Check if ML predictor is available and trained"""
        # This would check if ML model is loaded and ready
        return self._policy_performance[EvictionPolicy.ML_PREDICTED] > 0.1
