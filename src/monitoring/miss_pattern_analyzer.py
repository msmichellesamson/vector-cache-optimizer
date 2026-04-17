"""Cache miss pattern analyzer with ML-based miss prediction."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MissPattern:
    """Cache miss pattern data."""
    key_prefix: str
    miss_count: int
    miss_rate: float
    time_pattern: List[int]  # Hour of day distribution
    prediction_score: float

class MissPatternAnalyzer:
    """Analyzes cache miss patterns and predicts future misses."""
    
    def __init__(self, window_hours: int = 24, min_samples: int = 10):
        self.window_hours = window_hours
        self.min_samples = min_samples
        self.miss_history: deque = deque(maxlen=10000)
        self.key_patterns: Dict[str, List[datetime]] = defaultdict(list)
        self.hourly_distribution = np.zeros(24)
        
    def record_miss(self, cache_key: str, timestamp: Optional[datetime] = None) -> None:
        """Record a cache miss for pattern analysis."""
        if timestamp is None:
            timestamp = datetime.utcnow()
            
        self.miss_history.append((cache_key, timestamp))
        
        # Track key prefix patterns (first 20 chars)
        key_prefix = cache_key[:20] if len(cache_key) > 20 else cache_key
        self.key_patterns[key_prefix].append(timestamp)
        
        # Update hourly distribution
        self.hourly_distribution[timestamp.hour] += 1
        
        # Clean old entries
        self._cleanup_old_entries()
        
    def analyze_patterns(self) -> List[MissPattern]:
        """Analyze current miss patterns and return top patterns."""
        if len(self.miss_history) < self.min_samples:
            return []
            
        patterns = []
        cutoff_time = datetime.utcnow() - timedelta(hours=self.window_hours)
        
        for key_prefix, timestamps in self.key_patterns.items():
            recent_misses = [ts for ts in timestamps if ts >= cutoff_time]
            
            if len(recent_misses) < 3:  # Skip infrequent patterns
                continue
                
            miss_count = len(recent_misses)
            
            # Calculate hourly distribution for this pattern
            pattern_hours = np.zeros(24)
            for ts in recent_misses:
                pattern_hours[ts.hour] += 1
                
            # Normalize to get distribution
            if pattern_hours.sum() > 0:
                pattern_hours = pattern_hours / pattern_hours.sum()
                
            # Calculate prediction score based on pattern regularity
            prediction_score = self._calculate_prediction_score(pattern_hours)
            
            # Calculate miss rate (misses per hour)
            miss_rate = miss_count / self.window_hours
            
            pattern = MissPattern(
                key_prefix=key_prefix,
                miss_count=miss_count,
                miss_rate=miss_rate,
                time_pattern=pattern_hours.tolist(),
                prediction_score=prediction_score
            )
            patterns.append(pattern)
            
        # Sort by prediction score (highest first)
        return sorted(patterns, key=lambda p: p.prediction_score, reverse=True)
        
    def predict_next_miss_window(self, key_prefix: str) -> Optional[Tuple[int, float]]:
        """Predict the next likely miss window (hour) for a key prefix."""
        if key_prefix not in self.key_patterns:
            return None
            
        recent_misses = self._get_recent_misses(key_prefix)
        if len(recent_misses) < 3:
            return None
            
        # Build hourly pattern
        pattern_hours = np.zeros(24)
        for ts in recent_misses:
            pattern_hours[ts.hour] += 1
            
        if pattern_hours.sum() == 0:
            return None
            
        # Normalize and find peak hours
        pattern_hours = pattern_hours / pattern_hours.sum()
        current_hour = datetime.utcnow().hour
        
        # Find next high-probability hour
        for offset in range(1, 25):
            hour = (current_hour + offset) % 24
            probability = pattern_hours[hour]
            
            if probability > 0.1:  # Significant probability threshold
                return hour, probability
                
        return None
        
    def _calculate_prediction_score(self, pattern_hours: np.ndarray) -> float:
        """Calculate how predictable a pattern is (0-1 score)."""
        if pattern_hours.sum() == 0:
            return 0.0
            
        # Higher entropy = less predictable
        entropy = -np.sum(pattern_hours * np.log(pattern_hours + 1e-10))
        max_entropy = np.log(24)  # Uniform distribution
        
        # Convert to predictability score (inverse of normalized entropy)
        predictability = 1.0 - (entropy / max_entropy)
        return max(0.0, min(1.0, predictability))
        
    def _get_recent_misses(self, key_prefix: str) -> List[datetime]:
        """Get recent misses for a key prefix."""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.window_hours)
        return [ts for ts in self.key_patterns[key_prefix] if ts >= cutoff_time]
        
    def _cleanup_old_entries(self) -> None:
        """Remove entries older than window."""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.window_hours)
        
        # Clean key patterns
        for key_prefix in list(self.key_patterns.keys()):
            self.key_patterns[key_prefix] = [
                ts for ts in self.key_patterns[key_prefix] if ts >= cutoff_time
            ]
            
            # Remove empty patterns
            if not self.key_patterns[key_prefix]:
                del self.key_patterns[key_prefix]
                
        logger.debug(f"Cleaned old entries, tracking {len(self.key_patterns)} patterns")