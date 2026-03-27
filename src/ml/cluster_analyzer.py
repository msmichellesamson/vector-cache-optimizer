"""Embedding clustering for cache locality optimization."""

import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ClusterInfo:
    """Information about an embedding cluster."""
    cluster_id: int
    centroid: np.ndarray
    size: int
    avg_hit_rate: float
    keys: List[str]

class ClusterAnalyzer:
    """Analyzes embedding clusters to optimize cache locality."""
    
    def __init__(self, n_clusters: int = 8, min_cluster_size: int = 10):
        self.n_clusters = n_clusters
        self.min_cluster_size = min_cluster_size
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self._clusters: Dict[int, ClusterInfo] = {}
        self._key_to_cluster: Dict[str, int] = {}
    
    def analyze_embeddings(self, 
                          embeddings: Dict[str, np.ndarray], 
                          hit_rates: Dict[str, float]) -> Dict[int, ClusterInfo]:
        """Cluster embeddings and analyze cache patterns."""
        if len(embeddings) < self.min_cluster_size:
            logger.warning(f"Not enough embeddings ({len(embeddings)}) for clustering")
            return {}
        
        try:
            keys = list(embeddings.keys())
            vectors = np.array([embeddings[key] for key in keys])
            
            # Perform clustering
            cluster_labels = self.kmeans.fit_predict(vectors)
            
            # Build cluster information
            clusters = {}
            for i in range(self.n_clusters):
                cluster_keys = [keys[j] for j, label in enumerate(cluster_labels) if label == i]
                if len(cluster_keys) < 2:  # Skip tiny clusters
                    continue
                
                cluster_hit_rates = [hit_rates.get(key, 0.0) for key in cluster_keys]
                avg_hit_rate = np.mean(cluster_hit_rates)
                
                clusters[i] = ClusterInfo(
                    cluster_id=i,
                    centroid=self.kmeans.cluster_centers_[i],
                    size=len(cluster_keys),
                    avg_hit_rate=avg_hit_rate,
                    keys=cluster_keys
                )
                
                # Update key-to-cluster mapping
                for key in cluster_keys:
                    self._key_to_cluster[key] = i
            
            self._clusters = clusters
            logger.info(f"Created {len(clusters)} clusters from {len(embeddings)} embeddings")
            return clusters
            
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            return {}
    
    def get_cluster_locality_score(self, key: str) -> float:
        """Get locality score for a key based on its cluster."""
        cluster_id = self._key_to_cluster.get(key)
        if cluster_id is None or cluster_id not in self._clusters:
            return 0.0
        
        cluster = self._clusters[cluster_id]
        # Higher score for clusters with high hit rates and good size
        size_factor = min(cluster.size / 50.0, 1.0)  # Normalize to max of 50
        return cluster.avg_hit_rate * size_factor
    
    def should_colocate_keys(self, key1: str, key2: str) -> bool:
        """Check if two keys should be co-located in cache."""
        cluster1 = self._key_to_cluster.get(key1)
        cluster2 = self._key_to_cluster.get(key2)
        return cluster1 is not None and cluster1 == cluster2
    
    def get_cluster_stats(self) -> Dict[str, float]:
        """Get clustering statistics for monitoring."""
        if not self._clusters:
            return {"clusters": 0, "avg_cluster_size": 0, "avg_hit_rate": 0}
        
        sizes = [c.size for c in self._clusters.values()]
        hit_rates = [c.avg_hit_rate for c in self._clusters.values()]
        
        return {
            "clusters": len(self._clusters),
            "avg_cluster_size": np.mean(sizes),
            "avg_hit_rate": np.mean(hit_rates),
            "max_cluster_size": max(sizes),
            "min_cluster_size": min(sizes)
        }