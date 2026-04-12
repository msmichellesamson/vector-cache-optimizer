"""Vector deduplication service for cache optimization."""

import numpy as np
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import logging

logger = logging.getLogger(__name__)

@dataclass
class VectorGroup:
    """Group of similar vectors."""
    representative: str  # Key of representative vector
    members: Set[str]
    centroid: np.ndarray
    similarity_threshold: float

class VectorDeduplicator:
    """Identifies and groups similar vectors for cache deduplication."""
    
    def __init__(self, similarity_threshold: float = 0.95):
        self.similarity_threshold = similarity_threshold
        self.vector_groups: Dict[str, VectorGroup] = {}
        self.key_to_group: Dict[str, str] = {}
        
    def add_vector(self, key: str, vector: np.ndarray) -> Optional[str]:
        """Add vector and return representative key if duplicate found."""
        try:
            # Check existing groups for similarity
            for group_id, group in self.vector_groups.items():
                similarity = cosine_similarity(
                    vector.reshape(1, -1),
                    group.centroid.reshape(1, -1)
                )[0][0]
                
                if similarity >= self.similarity_threshold:
                    # Add to existing group
                    group.members.add(key)
                    self.key_to_group[key] = group_id
                    
                    # Update centroid
                    group.centroid = self._update_centroid(
                        group.centroid, vector, len(group.members)
                    )
                    
                    logger.debug(f"Vector {key} added to group {group_id} (similarity: {similarity:.3f})")
                    return group.representative
            
            # Create new group
            group_id = self._generate_group_id(vector)
            self.vector_groups[group_id] = VectorGroup(
                representative=key,
                members={key},
                centroid=vector.copy(),
                similarity_threshold=self.similarity_threshold
            )
            self.key_to_group[key] = group_id
            
            logger.debug(f"Created new group {group_id} for vector {key}")
            return None
            
        except Exception as e:
            logger.error(f"Error adding vector {key}: {e}")
            return None
    
    def get_duplicates(self, key: str) -> Set[str]:
        """Get all duplicate keys for a given key."""
        group_id = self.key_to_group.get(key)
        if not group_id:
            return set()
        
        group = self.vector_groups[group_id]
        return group.members - {group.representative}
    
    def get_deduplication_stats(self) -> Dict[str, int]:
        """Get deduplication statistics."""
        total_vectors = sum(len(group.members) for group in self.vector_groups.values())
        unique_groups = len(self.vector_groups)
        duplicates = total_vectors - unique_groups
        
        return {
            "total_vectors": total_vectors,
            "unique_groups": unique_groups,
            "duplicates_found": duplicates,
            "deduplication_ratio": duplicates / total_vectors if total_vectors > 0 else 0.0
        }
    
    def _update_centroid(self, current_centroid: np.ndarray, new_vector: np.ndarray, group_size: int) -> np.ndarray:
        """Update group centroid with new vector."""
        # Running average: new_centroid = (old_centroid * (n-1) + new_vector) / n
        return ((current_centroid * (group_size - 1)) + new_vector) / group_size
    
    def _generate_group_id(self, vector: np.ndarray) -> str:
        """Generate unique group ID from vector."""
        vector_hash = hashlib.md5(vector.tobytes()).hexdigest()[:8]
        return f"group_{vector_hash}"
