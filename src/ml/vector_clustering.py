from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.cluster import DBSCAN
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ClusterMetrics:
    cluster_id: int
    size: int
    avg_similarity: float
    access_frequency: int
    
class VectorClustering:
    """Groups similar vectors to optimize cache locality and eviction policies."""
    
    def __init__(self, eps: float = 0.15, min_samples: int = 3):
        self.eps = eps
        self.min_samples = min_samples
        self.clustering_model = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        self.cluster_metrics: Dict[int, ClusterMetrics] = {}
        
    def cluster_vectors(self, vectors: List[np.ndarray], keys: List[str]) -> Dict[str, int]:
        """Cluster vectors and return key->cluster_id mapping."""
        if len(vectors) < self.min_samples:
            logger.warning(f"Insufficient vectors for clustering: {len(vectors)}")
            return {key: -1 for key in keys}
            
        try:
            vector_matrix = np.vstack(vectors)
            cluster_labels = self.clustering_model.fit_predict(vector_matrix)
            
            key_to_cluster = dict(zip(keys, cluster_labels))
            self._update_cluster_metrics(vectors, cluster_labels, keys)
            
            logger.info(f"Clustered {len(vectors)} vectors into {len(set(cluster_labels))} clusters")
            return key_to_cluster
            
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            return {key: -1 for key in keys}
            
    def _update_cluster_metrics(self, vectors: List[np.ndarray], labels: List[int], keys: List[str]):
        """Update metrics for each cluster."""
        cluster_data = {}
        
        for i, (vector, label, key) in enumerate(zip(vectors, labels, keys)):
            if label == -1:  # Noise point
                continue
                
            if label not in cluster_data:
                cluster_data[label] = {'vectors': [], 'keys': []}
            cluster_data[label]['vectors'].append(vector)
            cluster_data[label]['keys'].append(key)
            
        for cluster_id, data in cluster_data.items():
            cluster_vectors = data['vectors']
            cluster_size = len(cluster_vectors)
            
            # Calculate average pairwise similarity
            similarities = []
            for i in range(len(cluster_vectors)):
                for j in range(i+1, len(cluster_vectors)):
                    sim = np.dot(cluster_vectors[i], cluster_vectors[j]) / (
                        np.linalg.norm(cluster_vectors[i]) * np.linalg.norm(cluster_vectors[j])
                    )
                    similarities.append(sim)
                    
            avg_similarity = np.mean(similarities) if similarities else 0.0
            
            self.cluster_metrics[cluster_id] = ClusterMetrics(
                cluster_id=cluster_id,
                size=cluster_size,
                avg_similarity=avg_similarity,
                access_frequency=0
            )
            
    def get_eviction_priority(self, cluster_id: int) -> float:
        """Get eviction priority score for cluster (higher = evict first)."""
        if cluster_id not in self.cluster_metrics:
            return 1.0  # High priority for unknown clusters
            
        metrics = self.cluster_metrics[cluster_id]
        
        # Lower similarity + lower access frequency = higher eviction priority
        priority = (1.0 - metrics.avg_similarity) * 0.6 + (1.0 / (metrics.access_frequency + 1)) * 0.4
        return min(priority, 1.0)
        
    def record_access(self, cluster_id: int):
        """Record access to update cluster frequency."""
        if cluster_id in self.cluster_metrics:
            self.cluster_metrics[cluster_id].access_frequency += 1