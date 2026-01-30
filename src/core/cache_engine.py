"""Core cache engine with ML-driven eviction policies."""
import asyncio
import json
import time
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import struct

import redis.asyncio as redis
import numpy as np
import torch
import torch.nn as nn
from prometheus_client import Counter, Histogram, Gauge
import structlog

from .exceptions import CacheError, EvictionError, VectorError
from .models import EvictionPredictor
from .metrics import CacheMetrics

logger = structlog.get_logger()

# Metrics
CACHE_OPERATIONS = Counter('cache_operations_total', 'Total cache operations', ['operation', 'result'])
CACHE_LATENCY = Histogram('cache_operation_duration_seconds', 'Cache operation latency', ['operation'])
CACHE_SIZE = Gauge('cache_size_bytes', 'Current cache size in bytes')
CACHE_ITEMS = Gauge('cache_items_total', 'Total items in cache')
EVICTION_DECISIONS = Counter('eviction_decisions_total', 'Eviction decisions', ['policy', 'decision'])

class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"
    LFU = "lfu"
    ML_PREDICTED = "ml_predicted"
    HYBRID = "hybrid"

@dataclass
class CacheItem:
    """Represents a cached vector item with metadata."""
    key: str
    vector: bytes
    embedding_dim: int
    timestamp: float
    access_count: int
    last_access: float
    size_bytes: int
    similarity_scores: List[float]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Redis storage."""
        return {
            'key': self.key,
            'vector': self.vector,
            'embedding_dim': self.embedding_dim,
            'timestamp': self.timestamp,
            'access_count': self.access_count,
            'last_access': self.last_access,
            'size_bytes': self.size_bytes,
            'similarity_scores': json.dumps(self.similarity_scores),
            'metadata': json.dumps(self.metadata)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheItem':
        """Create from Redis dictionary."""
        return cls(
            key=data['key'],
            vector=data['vector'],
            embedding_dim=int(data['embedding_dim']),
            timestamp=float(data['timestamp']),
            access_count=int(data['access_count']),
            last_access=float(data['last_access']),
            size_bytes=int(data['size_bytes']),
            similarity_scores=json.loads(data.get('similarity_scores', '[]')),
            metadata=json.loads(data.get('metadata', '{}'))
        )

class VectorCacheEngine:
    """High-performance vector cache with ML-driven eviction."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        max_memory_mb: int = 1024,
        eviction_policy: EvictionPolicy = EvictionPolicy.HYBRID,
        ml_model_path: Optional[str] = None,
        embedding_dim: int = 768,
        similarity_threshold: float = 0.85
    ):
        self.redis_url = redis_url
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.eviction_policy = eviction_policy
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        
        self.redis_client: Optional[redis.Redis] = None
        self.eviction_predictor: Optional[EvictionPredictor] = None
        self.metrics = CacheMetrics()
        
        # Internal state
        self._current_memory = 0
        self._access_patterns: Dict[str, List[float]] = {}
        self._similarity_cache: Dict[str, Dict[str, float]] = {}
        self._eviction_candidates: Set[str] = set()
        
        # Load ML model if provided
        if ml_model_path and eviction_policy in [EvictionPolicy.ML_PREDICTED, EvictionPolicy.HYBRID]:
            try:
                self.eviction_predictor = EvictionPredictor.load(ml_model_path)
                logger.info("Loaded ML eviction predictor", model_path=ml_model_path)
            except Exception as e:
                logger.warning("Failed to load ML model, falling back to LRU", error=str(e))
                self.eviction_policy = EvictionPolicy.LRU

    async def initialize(self) -> None:
        """Initialize Redis connection and cache state."""
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                decode_responses=False,
                health_check_interval=30,
                socket_keepalive=True,
                socket_keepalive_options={1: 1, 2: 3, 3: 5}
            )
            
            await self.redis_client.ping()
            await self._recover_cache_state()
            
            logger.info(
                "Cache engine initialized",
                redis_url=self.redis_url,
                max_memory_mb=self.max_memory_bytes // (1024 * 1024),
                eviction_policy=self.eviction_policy.value
            )
            
        except Exception as e:
            raise CacheError(f"Failed to initialize cache engine: {e}") from e

    async def close(self) -> None:
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()

    async def get(self, key: str) -> Optional[np.ndarray]:
        """Retrieve vector from cache."""
        start_time = time.time()
        
        try:
            # Get item from Redis
            cached_data = await self.redis_client.hgetall(f"cache:{key}")
            
            if not cached_data:
                CACHE_OPERATIONS.labels(operation='get', result='miss').inc()
                return None

            # Deserialize and update access patterns
            item = CacheItem.from_dict(cached_data)
            vector = self._deserialize_vector(item.vector, item.embedding_dim)
            
            # Update access metadata
            await self._update_access_metadata(key, item)
            
            # Update metrics
            CACHE_OPERATIONS.labels(operation='get', result='hit').inc()
            self.metrics.record_hit()
            
            return vector
            
        except Exception as e:
            CACHE_OPERATIONS.labels(operation='get', result='error').inc()
            logger.error("Cache get failed", key=key, error=str(e))
            raise CacheError(f"Failed to get key {key}: {e}") from e
            
        finally:
            CACHE_LATENCY.labels(operation='get').observe(time.time() - start_time)

    async def put(
        self, 
        key: str, 
        vector: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store vector in cache with eviction if needed."""
        start_time = time.time()
        
        try:
            if vector.shape[0] != self.embedding_dim:
                raise VectorError(f"Vector dimension {vector.shape[0]} doesn't match expected {self.embedding_dim}")

            # Serialize vector and calculate size
            vector_bytes = self._serialize_vector(vector)
            item_size = len(vector_bytes) + len(key.encode()) + 200  # Approximate metadata overhead

            # Check if eviction is needed
            if await self._needs_eviction(item_size):
                await self._perform_eviction(item_size)

            # Create cache item
            now = time.time()
            similarity_scores = await self._compute_similarity_scores(key, vector)
            
            item = CacheItem(
                key=key,
                vector=vector_bytes,
                embedding_dim=self.embedding_dim,
                timestamp=now,
                access_count=1,
                last_access=now,
                size_bytes=item_size,
                similarity_scores=similarity_scores,
                metadata=metadata or {}
            )

            # Store in Redis
            pipe = self.redis_client.pipeline()
            pipe.hset(f"cache:{key}", mapping=item.to_dict())
            pipe.zadd("cache:lru", {key: now})
            pipe.zadd("cache:lfu", {key: 1})
            pipe.incr("cache:size", item_size)
            await pipe.execute()

            # Update internal state
            self._current_memory += item_size
            CACHE_SIZE.set(self._current_memory)
            CACHE_ITEMS.inc()
            
            CACHE_OPERATIONS.labels(operation='put', result='success').inc()
            logger.debug("Vector cached", key=key, size_bytes=item_size)
            
            return True
            
        except Exception as e:
            CACHE_OPERATIONS.labels(operation='put', result='error').inc()
            logger.error("Cache put failed", key=key, error=str(e))
            raise CacheError(f"Failed to put key {key}: {e}") from e
            
        finally:
            CACHE_LATENCY.labels(operation='put').observe(time.time() - start_time)

    async def delete(self, key: str) -> bool:
        """Remove item from cache."""
        try:
            # Get item size before deletion
            cached_data = await self.redis_client.hgetall(f"cache:{key}")
            if not cached_data:
                return False

            item_size = int(cached_data.get(b'size_bytes', 0))

            # Remove from Redis
            pipe = self.redis_client.pipeline()
            pipe.delete(f"cache:{key}")
            pipe.zrem("cache:lru", key)
            pipe.zrem("cache:lfu", key)
            pipe.decrby("cache:size", item_size)
            results = await pipe.execute()

            if results[0]:  # If deletion was successful
                self._current_memory -= item_size
                CACHE_SIZE.set(self._current_memory)
                CACHE_ITEMS.dec()
                
                # Clean up internal state
                self._access_patterns.pop(key, None)
                self._similarity_cache.pop(key, None)
                self._eviction_candidates.discard(key)
                
                logger.debug("Vector deleted", key=key, freed_bytes=item_size)
                return True

            return False
            
        except Exception as e:
            logger.error("Cache delete failed", key=key, error=str(e))
            raise CacheError(f"Failed to delete key {key}: {e}") from e

    async def similarity_search(
        self, 
        query_vector: np.ndarray, 
        top_k: int = 10,
        threshold: float = None
    ) -> List[Tuple[str, float]]:
        """Find similar vectors in cache."""
        start_time = time.time()
        threshold = threshold or self.similarity_threshold
        
        try:
            # Get all cached vectors
            keys = await self.redis_client.zrange("cache:lru", 0, -1)
            if not keys:
                return []

            similarities = []
            
            for key_bytes in keys:
                key = key_bytes.decode() if isinstance(key_bytes, bytes) else key_bytes
                cached_data = await self.redis_client.hgetall(f"cache:{key}")
                
                if cached_data:
                    item = CacheItem.from_dict(cached_data)
                    cached_vector = self._deserialize_vector(item.vector, item.embedding_dim)
                    
                    # Compute cosine similarity
                    similarity = self._cosine_similarity(query_vector, cached_vector)
                    
                    if similarity >= threshold:
                        similarities.append((key, float(similarity)))

            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            result = similarities[:top_k]
            
            logger.debug(
                "Similarity search completed",
                query_dim=query_vector.shape[0],
                candidates=len(keys),
                matches=len(result),
                threshold=threshold
            )
            
            return result
            
        except Exception as e:
            logger.error("Similarity search failed", error=str(e))
            raise CacheError(f"Similarity search failed: {e}") from e
            
        finally:
            CACHE_LATENCY.labels(operation='similarity_search').observe(time.time() - start_time)

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            total_items = await self.redis_client.zcard("cache:lru")
            total_size = await self.redis_client.get("cache:size") or 0
            
            return {
                'total_items': total_items,
                'total_size_bytes': int(total_size),
                'memory_usage_percent': (int(total_size) / self.max_memory_bytes) * 100,
                'eviction_policy': self.eviction_policy.value,
                'hit_rate': self.metrics.hit_rate,
                'avg_access_time': self.metrics.avg_access_time,
                'eviction_count': self.metrics.eviction_count
            }
        except Exception as e:
            logger.error("Failed to get cache stats", error=str(e))
            return {}

    async def _needs_eviction(self, new_item_size: int) -> bool:
        """Check if eviction is needed for new item."""
        current_size = await self.redis_client.get("cache:size") or 0
        return int(current_size) + new_item_size > self.max_memory_bytes

    async def _perform_eviction(self, required_space: int) -> None:
        """Perform cache eviction based on configured policy."""
        start_time = time.time()
        
        try:
            evicted_space = 0
            evicted_count = 0
            
            while evicted_space < required_space:
                if self.eviction_policy == EvictionPolicy.LRU:
                    candidate = await self._get_lru_candidate()
                elif self.eviction_policy == EvictionPolicy.LFU:
                    candidate = await self._get_lfu_candidate()
                elif self.eviction_policy == EvictionPolicy.ML_PREDICTED:
                    candidate = await self._get_ml_candidate()
                elif self.eviction_policy == EvictionPolicy.HYBRID:
                    candidate = await self._get_hybrid_candidate()
                else:
                    candidate = await self._get_lru_candidate()

                if not candidate:
                    break

                # Get item size before deletion
                cached_data = await self.redis_client.hgetall(f"cache:{candidate}")
                if cached_data:
                    item_size = int(cached_data.get(b'size_bytes', 0))
                    
                    if await self.delete(candidate):
                        evicted_space += item_size
                        evicted_count += 1
                        
                        EVICTION_DECISIONS.labels(
                            policy=self.eviction_policy.value,
                            decision='evicted'
                        ).inc()

            self.metrics.record_eviction(evicted_count)
            
            logger.info(
                "Eviction completed",
                policy=self.eviction_policy.value,
                evicted_items=evicted_count,
                freed_bytes=evicted_space,
                duration_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            logger.error("Eviction failed", error=str(e))
            raise EvictionError(f"Cache eviction failed: {e}") from e

    async def _get_lru_candidate(self) -> Optional[str]:
        """Get least recently used item."""
        result = await self.redis_client.zrange("cache:lru", 0, 0)
        return result[0].decode() if result else None

    async def _get_lfu_candidate(self) -> Optional[str]:
        """Get least frequently used item."""
        result = await self.redis_client.zrange("cache:lfu", 0, 0)
        return result[0].decode() if result else None

    async def _get_ml_candidate(self) -> Optional[str]:
        """Get ML-predicted eviction candidate."""
        if not self.eviction_predictor:
            return await self._get_lru_candidate()

        try:
            # Get cache items for ML prediction
            keys = await self.redis_client.zrange("cache:lru", 0, 99)  # Sample top 100
            candidates = []
            
            for key_bytes in keys:
                key = key_bytes.decode() if isinstance(key_bytes, bytes) else key_bytes
                cached_data = await self.redis_client.hgetall(f"cache:{key}")
                
                if cached_data:
                    item = CacheItem.from_dict(cached_data)
                    features = self._extract_features(item)
                    candidates.append((key, features))

            if not candidates:
                return None

            # Predict eviction probabilities
            features_tensor = torch.tensor([feat for _, feat in candidates], dtype=torch.float32)
            
            with torch.no_grad():
                probabilities = self.eviction_predictor(features_tensor)
                best_idx = torch.argmax(probabilities).item()

            return candidates[best_idx][0]
            
        except Exception as e:
            logger.warning("ML candidate selection failed, falling back to LRU", error=str(e))
            return await self._get_lru_candidate()

    async def _get_hybrid_candidate(self) -> Optional[str]:
        """Get hybrid policy candidate (ML + traditional)."""
        # Weight 70% ML prediction, 30% LRU
        if self.eviction_predictor and np.random.random() < 0.7:
            return await self._get_ml_candidate()
        else:
            return await self._get_lru_candidate()

    async def _update_access_metadata(self, key: str, item: CacheItem) -> None:
        """Update access patterns for item."""
        now = time.time()
        
        # Update Redis metadata
        pipe = self.redis_client.pipeline()
        pipe.hset(f"cache:{key}", "access_count", item.access_count + 1)
        pipe.hset(f"cache:{key}", "last_access", now)
        pipe.zadd("cache:lru", {key: now})
        pipe.zincrby("cache:lfu", 1, key)
        await pipe.execute()

        # Update internal access patterns
        if key not in self._access_patterns:
            self._access_patterns[key] = []
        
        self._access_patterns[key].append(now)
        
        # Keep only recent access patterns (last 100)
        if len(self._access_patterns[key]) > 100:
            self._access_patterns[key] = self._access_patterns[key][-100:]

    async def _compute_similarity_scores(self, key: str, vector: np.ndarray) -> List[float]:
        """Compute similarity scores with existing cache items."""
        try:
            # Sample some existing vectors for similarity computation
            sample_keys = await self.redis_client.zrange("cache:lru", -10, -1)  # Last 10 items
            scores = []
            
            for sample_key_bytes in sample_keys:
                sample_key = sample_key_bytes.decode() if isinstance(sample_key_bytes, bytes) else sample_key_bytes
                if sample_key != key:
                    cached_data = await self.redis_client.hgetall(f"cache:{sample_key}")
                    if cached_data:
                        item = CacheItem.from_dict(cached_data)
                        cached_vector = self._deserialize_vector(item.vector, item.embedding_dim)
                        similarity = self._cosine_similarity(vector, cached_vector)
                        scores.append(float(similarity))

            return scores[-5:]  # Keep only top 5 similarity scores
            
        except Exception as e:
            logger.warning("Failed to compute similarity scores", key=key, error=str(e))
            return []

    async def _recover_cache_state(self) -> None:
        """Recover cache state from Redis on startup."""
        try:
            current_size = await self.redis_client.get("cache:size")
            self._current_memory = int(current_size) if current_size else 0
            
            item_count = await self.redis_client.zcard("cache:lru")
            
            CACHE_SIZE.set(self._current_memory)
            CACHE_ITEMS.set(item_count)
            
            logger.info(
                "Cache state recovered",
                items=item_count,
                size_mb=self._current_memory // (1024 * 1024)
            )
            
        except Exception as e:
            logger.warning("Failed to recover cache state", error=str(e))

    def _serialize_vector(self, vector: np.ndarray) -> bytes:
        """Serialize numpy vector to bytes."""
        return vector.astype(np.float32).tobytes()

    def _deserialize_vector(self, vector_bytes: bytes, dim: int) -> np.ndarray:
        """Deserialize bytes to numpy vector."""
        return np.frombuffer(vector_bytes, dtype=np.float32).reshape(-1)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)

    def _extract_features(self, item: CacheItem) -> List[float]:
        """Extract features for ML eviction prediction."""
        now = time.time()
        
        # Time-based features
        age = now - item.timestamp
        time_since_access = now - item.last_access
        
        # Access pattern features
        access_frequency = item.access_count / max(age / 3600, 0.1)  # accesses per hour
        
        # Size features
        size_ratio = item.size_bytes / self.max_memory_bytes
        
        # Similarity features
        avg_similarity = np.mean(item.similarity_scores) if item.similarity_scores else 0.0
        max_similarity = max(item.similarity_scores) if item.similarity_scores else 0.0
        
        return [
            age / 3600,  # age in hours
            time_since_access / 3600,  # hours since last access
            access_frequency,
            size_ratio,
            avg_similarity,
            max_similarity,
            float(len(item.metadata)),  # metadata complexity
            float(item.embedding_dim / 1000)  # normalized embedding dimension
        ]