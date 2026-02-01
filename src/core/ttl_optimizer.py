import time
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class AccessPattern:
    """Track access patterns for cache keys"""
    last_access: float
    access_count: int
    access_intervals: list[float]
    
    def add_access(self) -> None:
        current_time = time.time()
        if self.last_access > 0:
            interval = current_time - self.last_access
            self.access_intervals.append(interval)
            # Keep only last 10 intervals for calculation
            self.access_intervals = self.access_intervals[-10:]
        
        self.last_access = current_time
        self.access_count += 1

class TTLOptimizer:
    """ML-driven TTL optimization based on access patterns"""
    
    def __init__(self, base_ttl: int = 3600, min_ttl: int = 60, max_ttl: int = 86400):
        self.base_ttl = base_ttl
        self.min_ttl = min_ttl
        self.max_ttl = max_ttl
        self.access_patterns: Dict[str, AccessPattern] = defaultdict(
            lambda: AccessPattern(0, 0, [])
        )
    
    def record_access(self, key: str) -> None:
        """Record access for TTL calculation"""
        self.access_patterns[key].add_access()
    
    def calculate_optimal_ttl(self, key: str) -> int:
        """Calculate optimal TTL based on access patterns"""
        pattern = self.access_patterns.get(key)
        if not pattern or pattern.access_count < 2:
            return self.base_ttl
        
        # Calculate access frequency (accesses per hour)
        time_span = time.time() - (pattern.last_access - sum(pattern.access_intervals[-5:]))
        if time_span <= 0:
            return self.base_ttl
            
        frequency = pattern.access_count / (time_span / 3600)
        
        # Higher frequency = longer TTL
        if frequency > 10:  # Very frequent
            multiplier = 2.0
        elif frequency > 1:  # Frequent
            multiplier = 1.5
        elif frequency > 0.1:  # Moderate
            multiplier = 1.0
        else:  # Infrequent
            multiplier = 0.5
        
        optimal_ttl = int(self.base_ttl * multiplier)
        return max(self.min_ttl, min(optimal_ttl, self.max_ttl))
    
    def get_stats(self) -> Dict[str, int]:
        """Get optimization statistics"""
        return {
            "tracked_keys": len(self.access_patterns),
            "total_accesses": sum(p.access_count for p in self.access_patterns.values())
        }