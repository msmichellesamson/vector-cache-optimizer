"""Embedding similarity analyzer for cache optimization."""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class SimilarityCluster:
    """Represents a cluster of similar embeddings."""
    center: np.ndarray
    members: List[str]
    avg_hit_rate: float
    last_accessed: float


class SimilarityAnalyzer:
    """Analyzes embedding similarities to optimize cache placement."""
    
    def __init__(self, similarity_threshold: float = 0.85, max_clusters: int = 100):
        self.similarity_threshold = similarity_threshold
        self.max_clusters = max_clusters
        self.clusters: Dict[str, SimilarityCluster] = {}
        
    def analyze_embedding(self, key: str, embedding: np.ndarray, 
                         hit_rate: float, timestamp: float) -> Optional[str]:
        """Analyze embedding and return cluster ID if similar cluster exists."""
        try:
            if len(self.clusters) == 0:
                cluster_id = f"cluster_{len(self.clusters)}"
                self.clusters[cluster_id] = SimilarityCluster(
                    center=embedding.copy(),
                    members=[key],
                    avg_hit_rate=hit_rate,
                    last_accessed=timestamp
                )
                return cluster_id
                
            # Find most similar cluster
            best_cluster = None
            best_similarity = 0.0
            
            for cluster_id, cluster in self.clusters.items():
                similarity = cosine_similarity(
                    embedding.reshape(1, -1),
                    cluster.center.reshape(1, -1)
                )[0][0]
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster = cluster_id
                    
            # Add to existing cluster or create new one
            if best_similarity >= self.similarity_threshold and best_cluster:
                self._update_cluster(best_cluster, key, embedding, hit_rate, timestamp)
                return best_cluster
            elif len(self.clusters) < self.max_clusters:
                cluster_id = f"cluster_{len(self.clusters)}"
                self.clusters[cluster_id] = SimilarityCluster(
                    center=embedding.copy(),
                    members=[key],
                    avg_hit_rate=hit_rate,
                    last_accessed=timestamp
                )
                return cluster_id
                
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing embedding for key {key}: {e}")
            return None
    
    def _update_cluster(self, cluster_id: str, key: str, embedding: np.ndarray,
                       hit_rate: float, timestamp: float) -> None:
        """Update cluster with new member."""
        cluster = self.clusters[cluster_id]
        
        # Update center (weighted average)
        alpha = 0.1  # Learning rate
        cluster.center = (1 - alpha) * cluster.center + alpha * embedding
        
        # Update metrics
        cluster.members.append(key)
        n = len(cluster.members)
        cluster.avg_hit_rate = ((n - 1) * cluster.avg_hit_rate + hit_rate) / n
        cluster.last_accessed = max(cluster.last_accessed, timestamp)
    
    def get_cluster_priority(self, cluster_id: str) -> float:
        """Calculate eviction priority for cluster (higher = keep longer)."""
        if cluster_id not in self.clusters:
            return 0.0
            
        cluster = self.clusters[cluster_id]
        # Priority based on hit rate and recency
        return cluster.avg_hit_rate * 0.7 + (cluster.last_accessed / 1000000) * 0.3
    
    def get_cluster_stats(self) -> Dict[str, Dict]:
        """Get statistics for all clusters."""
        return {
            cluster_id: {
                "member_count": len(cluster.members),
                "avg_hit_rate": cluster.avg_hit_rate,
                "last_accessed": cluster.last_accessed
            }
            for cluster_id, cluster in self.clusters.items()
        }
