from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class CacheAccessPattern:
    key_hash: str
    last_access: datetime
    access_count: int
    embedding_size: int
    query_similarity: float = 0.0

class HitPredictor:
    """Predicts cache hit probability using access patterns and embedding similarity."""
    
    def __init__(self, history_window: int = 1000):
        self.history_window = history_window
        self.access_history: Dict[str, CacheAccessPattern] = {}
        self.temporal_weights = np.exp(-np.arange(24) / 6)  # 24h decay, 6h half-life
        
    def record_access(self, key_hash: str, embedding_size: int, similarity: float = 0.0) -> None:
        """Record cache access for pattern learning."""
        now = datetime.utcnow()
        
        if key_hash in self.access_history:
            pattern = self.access_history[key_hash]
            pattern.last_access = now
            pattern.access_count += 1
            pattern.query_similarity = max(pattern.query_similarity, similarity)
        else:
            self.access_history[key_hash] = CacheAccessPattern(
                key_hash=key_hash,
                last_access=now,
                access_count=1,
                embedding_size=embedding_size,
                query_similarity=similarity
            )
            
        self._cleanup_old_entries()
    
    def predict_hit_score(self, key_hash: str, current_similarity: float = 0.0) -> float:
        """Return hit probability score (0-1) for cache key."""
        if key_hash not in self.access_history:
            return 0.1  # Low score for unknown keys
            
        pattern = self.access_history[key_hash]
        now = datetime.utcnow()
        
        # Temporal score: higher for recent access
        time_diff = (now - pattern.last_access).total_seconds() / 3600  # hours
        temporal_score = np.exp(-time_diff / 6)  # 6h half-life
        
        # Frequency score: normalized by access count
        frequency_score = min(pattern.access_count / 10, 1.0)  # Cap at 10 accesses
        
        # Similarity score: how similar to previous queries
        similarity_score = max(pattern.query_similarity, current_similarity)
        
        # Size penalty: smaller embeddings more likely to be accessed
        size_score = max(0.1, 1.0 - (pattern.embedding_size / 10000))  # Penalty for >10k dims
        
        # Weighted combination
        final_score = (
            0.4 * temporal_score +
            0.3 * frequency_score +
            0.2 * similarity_score +
            0.1 * size_score
        )
        
        return min(final_score, 1.0)
    
    def get_top_candidates(self, limit: int = 100) -> List[tuple[str, float]]:
        """Return top cache candidates by hit prediction score."""
        candidates = []
        for key_hash in self.access_history:
            score = self.predict_hit_score(key_hash)
            candidates.append((key_hash, score))
            
        return sorted(candidates, key=lambda x: x[1], reverse=True)[:limit]
    
    def _cleanup_old_entries(self) -> None:
        """Remove old access patterns to prevent memory growth."""
        if len(self.access_history) <= self.history_window:
            return
            
        cutoff_time = datetime.utcnow() - timedelta(hours=48)
        old_keys = [
            key for key, pattern in self.access_history.items()
            if pattern.last_access < cutoff_time
        ]
        
        for key in old_keys[:len(old_keys) // 2]:  # Remove half of old entries
            del self.access_history[key]
            
        logger.info(f"Cleaned up {len(old_keys) // 2} old cache patterns")