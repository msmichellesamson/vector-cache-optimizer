import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class AccessPattern:
    vector_id: str
    access_times: List[datetime]
    frequency_score: float
    recency_score: float
    temporal_pattern: str  # 'regular', 'burst', 'declining'

class PatternLearner:
    """Learns access patterns for embedding vectors to improve cache decisions."""
    
    def __init__(self, window_hours: int = 24, min_accesses: int = 3):
        self.window_hours = window_hours
        self.min_accesses = min_accesses
        self.access_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.learned_patterns: Dict[str, AccessPattern] = {}
        
    def record_access(self, vector_id: str) -> None:
        """Record an access to a vector."""
        now = datetime.utcnow()
        self.access_history[vector_id].append(now)
        
        # Update pattern if we have enough data
        if len(self.access_history[vector_id]) >= self.min_accesses:
            self._update_pattern(vector_id)
    
    def _update_pattern(self, vector_id: str) -> None:
        """Update the learned pattern for a vector."""
        accesses = list(self.access_history[vector_id])
        cutoff = datetime.utcnow() - timedelta(hours=self.window_hours)
        recent_accesses = [a for a in accesses if a >= cutoff]
        
        if len(recent_accesses) < 2:
            return
            
        # Calculate frequency score (accesses per hour)
        frequency_score = len(recent_accesses) / self.window_hours
        
        # Calculate recency score (inverse of hours since last access)
        hours_since_last = (datetime.utcnow() - recent_accesses[-1]).total_seconds() / 3600
        recency_score = 1.0 / (1.0 + hours_since_last)
        
        # Determine temporal pattern
        temporal_pattern = self._classify_temporal_pattern(recent_accesses)
        
        self.learned_patterns[vector_id] = AccessPattern(
            vector_id=vector_id,
            access_times=recent_accesses,
            frequency_score=frequency_score,
            recency_score=recency_score,
            temporal_pattern=temporal_pattern
        )
        
        logger.debug(f"Updated pattern for {vector_id}: freq={frequency_score:.3f}, "
                    f"recency={recency_score:.3f}, pattern={temporal_pattern}")
    
    def _classify_temporal_pattern(self, accesses: List[datetime]) -> str:
        """Classify the temporal access pattern."""
        if len(accesses) < 3:
            return 'insufficient_data'
            
        # Calculate intervals between accesses
        intervals = [(accesses[i+1] - accesses[i]).total_seconds() 
                    for i in range(len(accesses)-1)]
        
        if not intervals:
            return 'single_access'
            
        avg_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        # Regular pattern: low variance in intervals
        if std_interval < avg_interval * 0.3:
            return 'regular'
        
        # Burst pattern: many accesses in short time
        recent_accesses = sum(1 for a in accesses 
                            if (datetime.utcnow() - a).total_seconds() < 3600)
        if recent_accesses >= len(accesses) * 0.7:
            return 'burst'
            
        # Declining pattern: intervals getting longer
        if len(intervals) >= 3:
            recent_avg = np.mean(intervals[-2:])
            early_avg = np.mean(intervals[:2])
            if recent_avg > early_avg * 1.5:
                return 'declining'
                
        return 'irregular'
    
    def predict_future_access(self, vector_id: str, hours_ahead: float = 1.0) -> float:
        """Predict probability of access within the next N hours."""
        if vector_id not in self.learned_patterns:
            return 0.5  # Unknown pattern, neutral prediction
            
        pattern = self.learned_patterns[vector_id]
        
        # Base probability from frequency and recency
        base_prob = min(pattern.frequency_score * hours_ahead, 1.0)
        recency_weight = pattern.recency_score
        
        # Adjust based on temporal pattern
        pattern_multiplier = {
            'regular': 1.2,
            'burst': 0.8,  # Lower future probability after burst
            'declining': 0.6,
            'irregular': 1.0,
            'insufficient_data': 0.8,
            'single_access': 0.3
        }.get(pattern.temporal_pattern, 1.0)
        
        return min(base_prob * recency_weight * pattern_multiplier, 1.0)
    
    def get_cache_priority(self, vector_id: str) -> float:
        """Get cache priority score (0-1, higher = keep longer)."""
        prediction = self.predict_future_access(vector_id)
        
        if vector_id in self.learned_patterns:
            pattern = self.learned_patterns[vector_id]
            # Boost priority for regular patterns
            if pattern.temporal_pattern == 'regular':
                prediction *= 1.3
        
        return min(prediction, 1.0)